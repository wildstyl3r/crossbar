//! Interactive terminal chat REPL: multi-turn history, per-turn sampling
//! overrides, prompt-based tool calling and optional schema-constrained
//! output. Single-threaded by design; Ctrl+C cancels the running
//! generation.

use std::io::{BufRead, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context as _, Error as E, Result};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::chat::{Message, render_messages, render_turn_suffix};
use crate::generation::{
    GenerateOptions, GenerationOutcome, GenerationTiming, SamplingParams, StopReason,
    TextGeneration,
};
use crate::grammar::SchemaMasker;
use crate::tools::{self, ToolMode};

static CANCEL: AtomicBool = AtomicBool::new(false);

/// Maximum tool-call round trips per user turn.
const MAX_TOOL_ITERATIONS: usize = 4;

const DIM: &str = "\x1b[2m";
const CYAN: &str = "\x1b[36m";
const RED: &str = "\x1b[31m";
const RESET: &str = "\x1b[0m";

pub struct ReplConfig {
    pub sampling: SamplingParams,
    pub max_tokens: usize,
    pub tool_mode: ToolMode,
    pub system: Option<String>,
    pub max_positions: usize,
}

struct Session {
    system: Option<String>,
    tool_mode: ToolMode,
    tool_system: String,
    history: Vec<Message>,
    sampling: SamplingParams,
    max_tokens: usize,
    schema: Option<Value>,
    eos_tokens: Vec<u32>,
    color: bool,
    max_positions: usize,
    last_prompt_len: Option<usize>,
    last_generated: usize,
    /// Ids believed to be in the KV cache after the last generation
    /// together with the history length they cover; the next generation
    /// assembles its prompt on top of them when only turns were appended.
    next_prefix: Option<(Vec<u32>, usize)>,
    last_reused: usize,
}
impl Session {
    fn new(config: ReplConfig, eos_tokens: Vec<u32>, color: bool) -> Self {
        let tool_mode = config.tool_mode;
        let tool_system = tools::system_prompt(
            &tools::builtin_tools()
                .iter()
                .map(|t| t.name)
                .collect::<Vec<_>>(),
        );
        Self {
            system: config.system,
            tool_mode,
            tool_system,
            history: Vec::new(),
            sampling: config.sampling,
            max_tokens: config.max_tokens,
            schema: None,
            eos_tokens,
            color,
            max_positions: config.max_positions,
            last_prompt_len: None,
            last_generated: 0,
            next_prefix: None,
            last_reused: 0,
        }
    }

    fn render_prompt(&self, template: &str) -> Result<String> {
        let mut msgs: Vec<Message> = Vec::new();
        let mut system = String::new();
        if let Some(s) = &self.system {
            system.push_str(s);
        }
        if self.tool_mode != ToolMode::Off {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(&self.tool_system);
        }
        if !system.is_empty() {
            msgs.push(Message::system(system));
        }
        msgs.extend(self.history.iter().cloned());
        render_messages(template, &msgs)
    }

    /// The schema currently constraining output: an explicit user schema
    /// wins; `required` tool mode forces the tool-call envelope.
    fn active_schema(&self) -> Option<Value> {
        if self.schema.is_some() {
            return self.schema.clone();
        }
        (self.tool_mode == ToolMode::Required).then(tools::required_call_schema)
    }

    fn sampling_line(&self) -> String {
        format!(
            "temp {} · top-k {} · top-p {} · penalty {:.2}/{} · seed {} · max {} · tools {} · schema {} · window {}",
            self.sampling
                .temperature
                .map(|t| format!("{t:.2}"))
                .unwrap_or_else(|| "-".into()),
            self.sampling
                .top_k
                .map(|k| k.to_string())
                .unwrap_or_else(|| "-".into()),
            self.sampling
                .top_p
                .map(|p| format!("{p:.2}"))
                .unwrap_or_else(|| "-".into()),
            self.sampling.repeat_penalty,
            self.sampling.repeat_last_n,
            self.sampling
                .seed
                .map(|s| s.to_string())
                .unwrap_or_else(|| "-".into()),
            self.max_tokens,
            format!("{:?}", self.tool_mode).to_lowercase(),
            self.schema.as_ref().map(|_| "on").unwrap_or("off"),
            self.max_positions,
        )
    }
}

/// "1323/32768 (4.0%)" — context consumption display fragment.
fn ctx_fragment(used: usize, max: usize) -> String {
    let pct = if max > 0 {
        100.0 * used as f64 / max as f64
    } else {
        0.0
    };
    format!("{used}/{max} ({pct:.1}%)")
}

/// Rough per-message token count: the content encoded standalone plus a
/// flat turn-wrapper overhead (`<start_of_turn>role\n` + `<end_of_turn>\n`).
fn estimate_message_tokens(tokenizer: &Tokenizer, message: &Message) -> usize {
    let content = tokenizer
        .encode(message.content.as_str(), false)
        .map(|e| e.get_ids().len())
        .unwrap_or(0);
    content + 4
}

/// Streams decoded chunks to the terminal, but holds back output that
/// looks like a JSON tool call (which the TokenOutputStream delivers in
/// one piece at the end anyway) so the formatted event can be shown
/// instead of raw JSON.
struct StreamingSink {
    mode: StreamMode,
    suppressed: String,
}

enum StreamMode {
    Deciding,
    Live,
    Buffering,
}

impl StreamingSink {
    fn new() -> Self {
        Self {
            mode: StreamMode::Deciding,
            suppressed: String::new(),
        }
    }

    fn chunk(&mut self, chunk: &str) -> Result<()> {
        match self.mode {
            StreamMode::Deciding => {
                self.suppressed.push_str(chunk);
                let trimmed = self.suppressed.trim_start();
                if trimmed.is_empty() {
                    return Ok(());
                }
                if trimmed.starts_with('{')
                    || trimmed.starts_with("```")
                    || trimmed.starts_with("<tool")
                {
                    self.mode = StreamMode::Buffering;
                } else {
                    print!("{}", self.suppressed);
                    std::io::stdout().flush()?;
                    self.suppressed.clear();
                    self.mode = StreamMode::Live;
                }
            }
            StreamMode::Buffering => self.suppressed.push_str(chunk),
            StreamMode::Live => {
                print!("{chunk}");
                std::io::stdout().flush()?;
            }
        }
        Ok(())
    }
}

pub fn run(
    mut pipeline: TextGeneration,
    chat_template: String,
    eos_tokens: Vec<u32>,
    config: ReplConfig,
) -> Result<()> {
    let color = std::io::stdout().is_terminal();
    if let Err(e) = ctrlc::set_handler(|| CANCEL.store(true, Ordering::SeqCst)) {
        eprintln!("warning: could not install the Ctrl+C handler: {e}");
    }

    let mut session = Session::new(config, eos_tokens.clone(), color);
    println!(
        "{}",
        styled(
            color,
            DIM,
            "crossbar chat — /help for commands, Ctrl+C cancels a generation, /quit exits"
        )
    );
    if let Some(system) = &session.system {
        println!("{}", styled(color, DIM, &format!("system: {system}")));
    }
    println!("{}", styled(color, DIM, &session.sampling_line()));

    let stdin = std::io::stdin();
    let mut prompt_interrupts = 0usize;
    loop {
        print!("{} ", styled(color, CYAN, "you>"));
        std::io::stdout().flush()?;
        let Some(line) = read_multiline(&stdin)? else {
            println!();
            break;
        };
        if CANCEL.swap(false, Ordering::SeqCst) {
            prompt_interrupts += 1;
            if prompt_interrupts >= 2 {
                break;
            }
            println!(
                "{}",
                styled(
                    color,
                    DIM,
                    "^C interrupted — press Ctrl+C again or /quit to exit"
                )
            );
            continue;
        }
        prompt_interrupts = 0;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("//") {
            session.chat_turn(&mut pipeline, &chat_template, rest.trim())?;
            continue;
        }
        if trimmed.starts_with('/') {
            match session.command(trimmed, pipeline.tokenizer()) {
                Ok(true) => break,
                Ok(false) => {}
                Err(e) => println!("{}", styled(color, RED, &format!("error: {e:#}"))),
            }
            continue;
        }
        if let Err(e) = session.chat_turn(&mut pipeline, &chat_template, trimmed) {
            println!("{}", styled(color, RED, &format!("error: {e:#}")));
            session
                .history
                .truncate(session.history.len().saturating_sub(1));
        }
    }
    Ok(())
}

impl Session {
    fn chat_turn(
        &mut self,
        pipeline: &mut TextGeneration,
        template: &str,
        user_text: &str,
    ) -> Result<()> {
        let turn_start = self.history.len();
        self.history.push(Message::user(user_text));
        let mut transient: Vec<usize> = Vec::new();
        let mut iteration = 0usize;
        loop {
            CANCEL.store(false, Ordering::SeqCst);
            let mut opts = GenerateOptions {
                sampling: self.sampling.clone(),
                max_tokens: self.max_tokens,
                cancel: Some(&CANCEL),
                cached_prefix: None,
                ..Default::default()
            };
            if let Some(schema) = self.active_schema() {
                let masker =
                    SchemaMasker::new(pipeline.tokenizer(), &schema, self.eos_tokens.clone())
                        .context("failed to build the schema masker")?;
                opts.filter = Some(Box::new(masker));
            }
            let sink = std::rc::Rc::new(std::cell::RefCell::new(StreamingSink::new()));
            let sink_for_stream = sink.clone();
            opts.stream = Some(Box::new(move |chunk: &str| {
                sink_for_stream.borrow_mut().chunk(chunk)
            }));
            // Prompt assembly: with a retained cache whose covered history
            // still lines up with the (append-only) history, build the
            // prompt at the token level — the cached ids plus the suffix
            // rendered by the template — so the prefix always matches and
            // the conversation is not re-prefilled. Otherwise render fully
            // (the token compare inside generate still guards).
            let (mut outcome, timing) = match self.next_prefix.take() {
                Some((prefix_ids, covered)) if covered < self.history.len() => {
                    let suffix = render_turn_suffix(template, &self.history[covered..])?;
                    let suffix_ids = pipeline
                        .tokenizer()
                        .encode(suffix.as_str(), false)
                        .map_err(E::msg)?
                        .get_ids()
                        .to_vec();
                    let mut prompt_ids = prefix_ids.clone();
                    prompt_ids.extend(suffix_ids);
                    opts.cached_prefix = Some(prefix_ids);
                    pipeline.generate_from(prompt_ids, opts)?
                }
                cached => {
                    opts.cached_prefix = cached.map(|(ids, _)| ids);
                    let prompt = self.render_prompt(template)?;
                    pipeline.generate(&prompt, opts)?
                }
            };
            CANCEL.store(false, Ordering::SeqCst);
            let text = tools::strip_tool_blocks(&outcome.text);
            if outcome.reason == StopReason::Cancelled {
                println!();
                println!("{}", styled(self.color, DIM, "cancelled"));
                self.history.truncate(turn_start);
                return Ok(());
            }
            // The cached ids cover the whole history plus the reply that
            // is pushed right below — hence the +1.
            self.next_prefix = Some((
                std::mem::take(&mut outcome.context_ids),
                self.history.len() + 1,
            ));
            self.last_prompt_len = Some(timing.prompt_len);
            self.last_reused = timing.reused_prefix;
            self.last_generated = outcome.tokens;
            println!();
            print_stats(&outcome, &timing, self.max_positions, self.color);
            let used = timing.prompt_len + outcome.tokens;
            if used * 10 >= self.max_positions * 9 {
                println!(
                    "{}",
                    styled(
                        self.color,
                        RED,
                        &format!(
                            "warning: context {} — /reset to start fresh",
                            ctx_fragment(used, self.max_positions)
                        )
                    )
                );
            }

            if self.schema.is_some() {
                match serde_json::from_str::<Value>(&text) {
                    Ok(v) => println!(
                        "{}",
                        styled(
                            self.color,
                            CYAN,
                            &serde_json::to_string_pretty(&v).unwrap_or_else(|_| text.clone())
                        )
                    ),
                    Err(_) => println!("{text}"),
                }
                self.history.push(Message::assistant(text));
                break;
            }

            self.history.push(Message::assistant(text.clone()));
            if self.tool_mode == ToolMode::Off {
                break;
            }

            match tools::extract_tool_call(&text) {
                Some(call) => {
                    println!(
                        "{}",
                        styled(
                            self.color,
                            CYAN,
                            &format!(
                                "→ {}({})",
                                call.name,
                                serde_json::to_string(&call.arguments).unwrap_or_default()
                            )
                        )
                    );
                    let mut result = match tools::execute(&call) {
                        Ok(r) => r,
                        Err(e) => {
                            if tools::find_tool(&call.name).is_none() {
                                tools::unknown_tool_error(&call.name)
                            } else {
                                format!("Error: {e:#}")
                            }
                        }
                    };
                    for line in result.lines() {
                        println!("{} {line}", styled(self.color, CYAN, "←"));
                    }
                    if iteration + 1 == MAX_TOOL_ITERATIONS {
                        result.push_str(
                            "\n[System] Tool call limit reached. Answer the user now in plain text without calling any more tools.",
                        );
                    }
                    self.history.push(Message::tool_result(&call.name, result));
                    iteration += 1;
                    if iteration > MAX_TOOL_ITERATIONS {
                        break;
                    }
                }
                None => {
                    let suppressed = tools::strip_tool_blocks(&sink.borrow().suppressed);
                    if !suppressed.is_empty() {
                        println!("{}", styled(self.color, DIM, &suppressed));
                    }
                    if self.tool_mode == ToolMode::Required && iteration == 0 {
                        transient.push(self.history.len());
                        self.history.push(Message::user(
                            "Your reply was not a valid tool call. Reply with ONLY the JSON tool call object ({\"name\": ..., \"arguments\": ...}) or answer the user in plain text."
                                .to_string(),
                        ));
                        iteration += 1;
                    } else {
                        break;
                    }
                }
            }
        }
        for index in transient.into_iter().rev() {
            self.history.remove(index);
        }
        Ok(())
    }

    /// Handle a `/command`; returns `true` when the REPL should quit.
    fn command(&mut self, line: &str, tokenizer: &Tokenizer) -> Result<bool> {
        let (cmd, rest) = line.split_once(' ').unwrap_or((line, ""));
        let rest = rest.trim();
        let value = |what: &str| -> Result<f64> {
            rest.parse::<f64>()
                .map_err(|_| anyhow::anyhow!("{what} needs a number, got {rest:?}"))
        };
        match cmd {
            "/help" => {
                println!("commands:");
                println!("  /system [text]     show or set the system prompt");
                println!("  /reset             clear the conversation history");
                println!("  /temp <f>          sampling temperature (0 = greedy)");
                println!("  /topk <n|off>      top-k sampling");
                println!("  /topp <p|off>      nucleus sampling");
                println!("  /penalty <f>       repeat penalty");
                println!("  /seed <n|off>      sampling seed");
                println!("  /max <n>           max tokens per generation");
                println!("  /ctx               context window consumption statistics");
                println!("  /tool <off|auto|required>  tool calling mode");
                println!("  /tools             list available tools");
                println!("  /schema <file|off> constrain output to a JSON schema");
                println!("  /save <file>       save the transcript as JSON lines");
                println!("  /quit              exit (Ctrl+C twice also works)");
                println!("a trailing \\ continues on the next line; //text sends text literally");
            }
            "/system" => {
                if rest.is_empty() {
                    match &self.system {
                        Some(s) => println!("{s}"),
                        None => println!("{}", styled(self.color, DIM, "(no system prompt)")),
                    }
                } else {
                    self.system = Some(rest.to_string());
                    self.next_prefix = None;
                    println!("{}", styled(self.color, DIM, "system prompt updated"));
                }
            }
            "/reset" => {
                self.history.clear();
                self.next_prefix = None;
                println!("{}", styled(self.color, DIM, "history cleared"));
            }
            "/temp" => {
                let v = value("--temp")?;
                self.sampling.temperature = Some(v.max(0.0));
            }
            "/topk" => {
                self.sampling.top_k = if rest == "off" {
                    None
                } else {
                    Some(
                        rest.parse::<usize>()
                            .context("/topk needs a number or 'off'")?,
                    )
                };
            }
            "/topp" => {
                self.sampling.top_p = if rest == "off" {
                    None
                } else {
                    Some(value("--topp")?)
                };
            }
            "/penalty" => {
                self.sampling.repeat_penalty = value("--penalty")? as f32;
            }
            "/seed" => {
                self.sampling.seed = if rest == "off" {
                    None
                } else {
                    Some(
                        rest.parse::<u64>()
                            .context("/seed needs a number or 'off'")?,
                    )
                };
            }
            "/max" => {
                self.max_tokens = rest
                    .parse::<usize>()
                    .context("/max needs a token count")?
                    .max(1);
            }
            "/tool" => {
                self.tool_mode = match rest {
                    "off" => ToolMode::Off,
                    "auto" => ToolMode::Auto,
                    "required" => ToolMode::Required,
                    other => anyhow::bail!("/tool needs off, auto or required, got {other:?}"),
                };
                self.next_prefix = None;
            }
            "/tools" => {
                for tool in tools::builtin_tools() {
                    println!("{} — {}", tool.name, tool.description);
                }
            }
            "/schema" => {
                if rest == "off" || rest.is_empty() {
                    self.schema = None;
                    println!("{}", styled(self.color, DIM, "schema off"));
                } else {
                    let text = std::fs::read_to_string(rest)
                        .with_context(|| format!("cannot read {rest}"))?;
                    let schema: Value =
                        serde_json::from_str(&text).context("schema is not valid JSON")?;
                    crate::grammar::validate_schema(&schema)?;
                    self.schema = Some(schema);
                    println!(
                        "{}",
                        styled(self.color, DIM, &format!("schema set from {rest}"))
                    );
                }
            }
            "/save" => {
                use std::io::BufWriter;
                let file = std::fs::File::create(rest).context("/save needs a writable path")?;
                let mut out = BufWriter::new(file);
                for message in &self.history {
                    let role = match &message.role {
                        crate::chat::MessageRole::System => "system".to_string(),
                        crate::chat::MessageRole::User => "user".to_string(),
                        crate::chat::MessageRole::Assistant => "assistant".to_string(),
                        crate::chat::MessageRole::ToolResult(name) => {
                            format!("tool_result:{name}")
                        }
                    };
                    serde_json::to_writer(
                        &mut out,
                        &serde_json::json!({ "role": role, "content": message.content }),
                    )?;
                    writeln!(out)?;
                }
                out.flush()?;
                println!(
                    "{}",
                    styled(
                        self.color,
                        DIM,
                        &format!("saved {} messages to {rest}", self.history.len())
                    )
                );
            }
            "/ctx" => {
                let max = self.max_positions;
                println!("context window: {max} tokens");
                match self.last_prompt_len {
                    Some(prompt_len) => {
                        let used = prompt_len + self.last_generated;
                        let free = max.saturating_sub(used);
                        let prefilled = prompt_len.saturating_sub(self.last_reused);
                        println!(
                            "last turn: reused {r} · prefilled {prefilled} + generated {} → next render ≈ {} · {free} free",
                            self.last_generated,
                            ctx_fragment(used, max),
                            r = self.last_reused,
                        );
                    }
                    None => {
                        println!("last turn: none yet — the first prompt sets the baseline")
                    }
                }
                if !self.history.is_empty() {
                    let parts: Vec<String> = self
                        .history
                        .iter()
                        .map(|m| {
                            let role = match &m.role {
                                crate::chat::MessageRole::System => "system".to_string(),
                                crate::chat::MessageRole::User => "user".to_string(),
                                crate::chat::MessageRole::Assistant => "assistant".to_string(),
                                crate::chat::MessageRole::ToolResult(name) => {
                                    format!("tool({name})")
                                }
                            };
                            format!("{role} ≈{}", estimate_message_tokens(tokenizer, m))
                        })
                        .collect();
                    println!("≈ history: {}", parts.join(" · "));
                }
            }
            "/quit" | "/exit" => return Ok(true),
            other => anyhow::bail!("unknown command {other:?} — /help lists commands"),
        }
        Ok(false)
    }
}

fn read_multiline(stdin: &std::io::Stdin) -> Result<Option<String>> {
    let mut buf = String::new();
    let mut first = true;
    loop {
        if !first {
            print!("{} ", styled(std::io::stdout().is_terminal(), CYAN, "..>"));
            std::io::stdout().flush()?;
        }
        first = false;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            return Ok(if buf.is_empty() { None } else { Some(buf) });
        }
        let trimmed = line.trim_end();
        match trimmed.strip_suffix('\\') {
            Some(head) => {
                buf.push_str(head.trim_end());
                buf.push('\n');
            }
            None => {
                buf.push_str(&line);
                return Ok(Some(buf));
            }
        }
    }
}

fn styled(color: bool, code: &str, text: &str) -> String {
    if color {
        format!("{code}{text}{RESET}")
    } else {
        text.to_string()
    }
}

fn print_stats(
    outcome: &GenerationOutcome,
    timing: &GenerationTiming,
    max_positions: usize,
    color: bool,
) {
    let tps = outcome.tokens as f64 / timing.total.as_secs_f64().max(1e-9);
    let ctx = ctx_fragment(timing.prompt_len + outcome.tokens, max_positions);
    let prefilled = timing.prompt_len.saturating_sub(timing.reused_prefix);
    println!(
        "{}",
        styled(
            color,
            DIM,
            &format!(
                "[{} tok · {tps:.1} tok/s · {} · ctx {ctx} · new {prefilled}]",
                outcome.tokens, outcome.reason
            )
        )
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctx_fragment_formats_used_max_and_percent() {
        assert_eq!(ctx_fragment(1323, 32768), "1323/32768 (4.0%)");
        assert_eq!(ctx_fragment(32768, 32768), "32768/32768 (100.0%)");
        assert_eq!(ctx_fragment(0, 0), "0/0 (0.0%)");
    }

    #[test]
    fn streaming_sink_decides_on_first_real_chunk() {
        let mut sink = StreamingSink::new();
        sink.chunk("  ").unwrap();
        assert!(matches!(sink.mode, StreamMode::Deciding));
        sink.chunk("Hello").unwrap();
        assert!(matches!(sink.mode, StreamMode::Live));
        assert!(sink.suppressed.is_empty());

        let mut sink = StreamingSink::new();
        sink.chunk("{\"na").unwrap();
        assert!(matches!(sink.mode, StreamMode::Buffering));
        sink.chunk("me\": 1}").unwrap();
        assert_eq!(sink.suppressed, "{\"name\": 1}");

        let mut sink = StreamingSink::new();
        sink.chunk("```json\n{}").unwrap();
        assert!(matches!(sink.mode, StreamMode::Buffering));
    }
}
