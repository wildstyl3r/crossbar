//! Built-in tool registry for prompt-based tool calling. Tools are declared
//! to the model through a system-prompt block; calls are extracted leniently
//! from the generated text as JSON objects `{"name": ..., "arguments": ...}`.

use anyhow::{Context as _, Result, bail};
use serde_json::Value;

/// When the tool loop runs during a REPL turn.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToolMode {
    /// Plain chat; generated text is never parsed for tool calls.
    #[default]
    Off,
    /// Parse the reply for a tool call and run it when one is found.
    Auto,
    /// Like auto, but a missing tool call triggers a corrective retry turn.
    Required,
}

pub struct Tool {
    pub name: &'static str,
    pub description: &'static str,
    /// JSON schema (object) of the arguments.
    pub parameters: fn() -> Value,
    pub run: fn(&Value) -> Result<String>,
}

impl Tool {
    fn spec_json(&self) -> Value {
        serde_json::json!({
            "name": self.name,
            "description": self.description,
            "parameters": (self.parameters)(),
        })
    }
}

pub fn builtin_tools() -> &'static [Tool] {
    static TOOLS: std::sync::OnceLock<Vec<Tool>> = std::sync::OnceLock::new();
    TOOLS.get_or_init(|| {
        vec![
            Tool {
                name: "current_time",
                description: "Returns the current UTC date and time.",
                parameters: || {
                    serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "required": [],
                    })
                },
                run: run_current_time,
            },
            Tool {
                name: "calculator",
                description: "Evaluates an arithmetic expression with digits and operators (+ - * / % ^ and parentheses).",
                parameters: || {
                    serde_json::json!({
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The arithmetic expression to evaluate, e.g. \"(2 + 3) * 4\".",
                            },
                        },
                        "required": ["expression"],
                    })
                },
                run: run_calculator,
            },
            Tool {
                name: "echo",
                description: "Returns its input text unchanged.",
                parameters: || {
                    serde_json::json!({
                        "type": "object",
                        "properties": {
                            "text": { "type": "string", "description": "Text to echo back." },
                        },
                        "required": ["text"],
                    })
                },
                run: run_echo,
            },
        ]
    })
}

pub fn find_tool(name: &str) -> Option<&'static Tool> {
    builtin_tools().iter().find(|t| t.name == name)
}

/// The tool-declaration block appended to the system prompt.
pub fn system_prompt(active: &[&'static str]) -> String {
    let mut out = String::new();
    out.push_str("You can call tools. Available tools:\n");
    for name in active {
        match find_tool(name) {
            Some(tool) => out.push_str(&format!(
                "<tool>{}</tool>\n",
                serde_json::to_string(&tool.spec_json()).unwrap_or_default()
            )),
            None => eprintln!("warning: unknown tool {name:?} requested, skipping"),
        }
    }
    out.push_str(
        "Rules for tool use:\n\
         - To call a tool, reply with ONLY one JSON object: {\"name\": \"<tool name>\", \"arguments\": {<tool parameters>}}\n\
         - When calling a tool, do not write anything else before or after the JSON object.\n\
         - Never repeat or copy the tool definitions, the <tool> lines or this instruction text.\n\
         - After a call you receive a message starting with \"[Tool result for ...]\". Then either call another tool or answer the user in plain text.\n\
         - If no tool is needed, answer in plain text without any JSON object.\n",
    );
    out
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Leniently find the first tool call in generated text: strips code fences
/// and prose, then tries every balanced JSON object until one parses as a
/// call — an object with only `name` (string) and optional `arguments`
/// (object, defaulted to `{}`). Tool-spec echoes (which also carry a
/// `name` but have `description`/`parameters`) are rejected.
pub fn extract_tool_call(text: &str) -> Option<ToolCall> {
    for candidate in balanced_json_objects(text) {
        let Ok(value) = serde_json::from_str::<Value>(candidate) else {
            continue;
        };
        let Some(obj) = value.as_object() else {
            continue;
        };
        if obj.keys().any(|k| k != "name" && k != "arguments") {
            continue;
        }
        let Some(name) = obj.get("name").and_then(Value::as_str) else {
            continue;
        };
        let arguments = obj
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        if !arguments.is_object() {
            continue;
        }
        return Some(ToolCall {
            name: name.to_string(),
            arguments,
        });
    }
    None
}

/// Remove `<tool>...</tool>` spec echoes the model may copy from the
/// system prompt.
pub fn strip_tool_blocks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<tool>") {
        out.push_str(&rest[..start]);
        rest = match rest[start..].find("</tool>") {
            Some(end) => &rest[start + end + "</tool>".len()..],
            None => "",
        };
    }
    out.push_str(rest);
    out.trim().to_string()
}

/// Execute a tool call against the registry.
pub fn execute(call: &ToolCall) -> Result<String> {
    let tool = find_tool(&call.name).with_context(|| format!("unknown tool {:?}", call.name))?;
    (tool.run)(&call.arguments)
}

/// The error line returned to the model when a call does not match the
/// registry (lists the valid names — small models copy them).
pub fn unknown_tool_error(name: &str) -> String {
    let names: Vec<&str> = builtin_tools().iter().map(|t| t.name).collect();
    format!(
        "Error: unknown tool {name:?}. Valid tool names: {}.",
        names.join(", ")
    )
}

/// Schema of the forced tool-call envelope for `required` mode: the name
/// is constrained to the registry enum, the arguments to the union of all
/// built-in tool properties (all required — a small model otherwise skips
/// them; each tool ignores the other tools' keys).
pub fn required_call_schema() -> Value {
    let mut props = serde_json::Map::new();
    for tool in builtin_tools() {
        let params = (tool.parameters)();
        if let Some(p) = params.get("properties").and_then(Value::as_object) {
            for (k, v) in p {
                props.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
    }
    let names: Vec<Value> = builtin_tools()
        .iter()
        .map(|t| Value::String(t.name.into()))
        .collect();
    let all_props: Vec<Value> = props.keys().map(|k| Value::String(k.clone())).collect();
    serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "enum": names },
            "arguments": {
                "type": "object",
                "properties": props,
                "required": all_props,
            },
        },
        "required": ["name", "arguments"],
    })
}

fn balanced_json_objects(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }
        match balanced_close(bytes, i) {
            Some(end) => {
                out.push(&text[i..=end]);
                i = end + 1;
            }
            // Unclosed fragment (typical small-model debris): skip this
            // brace and let the next one start a candidate.
            None => i += 1,
        }
    }
    out
}

/// End index of the balanced object starting at `start` (which must be a
/// `{`), honoring strings and escapes.
fn balanced_close(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut i = start;
    while i < bytes.len() {
        let b = bytes[i];
        i += 1;
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i - 1);
                }
            }
            _ => {}
        }
    }
    None
}

fn run_current_time(_args: &Value) -> Result<String> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("system clock is before the unix epoch")?
        .as_secs();
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (y, m, d) = civil_from_days(days);
    Ok(format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02} UTC",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    ))
}

/// Howard Hinnant's days-from-civil inverse (proleptic Gregorian).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn run_calculator(args: &Value) -> Result<String> {
    let expression = args
        .get("expression")
        .and_then(Value::as_str)
        .context("calculator: missing string argument \"expression\"")?;
    let value = eval_arithmetic(expression)?;
    Ok(format_number(value))
}

fn run_echo(args: &Value) -> Result<String> {
    let text = args
        .get("text")
        .and_then(Value::as_str)
        .context("echo: missing string argument \"text\"")?;
    Ok(text.to_string())
}

fn format_number(r: f64) -> String {
    if r.is_finite() && r.fract() == 0.0 && r.abs() < 9.007_199_254_740_992e15 {
        format!("{}", r as i64)
    } else {
        format!("{r}")
    }
}

/// Recursive-descent evaluator for + - * / % ^ with unary signs,
/// parentheses and decimal/exponent literals. Division or modulo by zero
/// and malformed input are errors.
pub fn eval_arithmetic(input: &str) -> Result<f64> {
    let mut p = Parser {
        s: input.as_bytes(),
        pos: 0,
    };
    let value = p.expr()?;
    p.skip_ws();
    if p.pos != p.s.len() {
        bail!("unexpected {:?} at position {}", p.s[p.pos] as char, p.pos);
    }
    Ok(value)
}

struct Parser<'a> {
    s: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn skip_ws(&mut self) {
        while self.pos < self.s.len() && self.s[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.s.get(self.pos).copied()
    }

    fn expect(&mut self, b: u8) -> Result<()> {
        match self.peek() {
            Some(x) if x == b => {
                self.pos += 1;
                Ok(())
            }
            _ => bail!("expected {:#}", b as char),
        }
    }

    fn expr(&mut self) -> Result<f64> {
        let mut value = self.term()?;
        loop {
            match self.peek() {
                Some(b'+') => {
                    self.pos += 1;
                    value += self.term()?;
                }
                Some(b'-') => {
                    self.pos += 1;
                    value -= self.term()?;
                }
                _ => return Ok(value),
            }
        }
    }

    fn term(&mut self) -> Result<f64> {
        let mut value = self.factor()?;
        loop {
            match self.peek() {
                Some(b'*') => {
                    self.pos += 1;
                    value *= self.factor()?;
                }
                Some(b'/') => {
                    self.pos += 1;
                    let rhs = self.factor()?;
                    if rhs == 0.0 {
                        bail!("division by zero");
                    }
                    value /= rhs;
                }
                Some(b'%') => {
                    self.pos += 1;
                    let rhs = self.factor()?;
                    if rhs == 0.0 {
                        bail!("modulo by zero");
                    }
                    value %= rhs;
                }
                _ => return Ok(value),
            }
        }
    }

    fn factor(&mut self) -> Result<f64> {
        match self.peek() {
            Some(b'+') => {
                self.pos += 1;
                self.factor()
            }
            Some(b'-') => {
                self.pos += 1;
                Ok(-self.factor()?)
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<f64> {
        let base = self.primary()?;
        if self.peek() == Some(b'^') {
            self.pos += 1;
            let exponent = self.factor()?;
            Ok(base.powf(exponent))
        } else {
            Ok(base)
        }
    }

    fn primary(&mut self) -> Result<f64> {
        if self.peek() == Some(b'(') {
            self.pos += 1;
            let value = self.expr()?;
            self.expect(b')')?;
            return Ok(value);
        }
        self.number()
    }

    fn number(&mut self) -> Result<f64> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.s.len() && self.s[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos < self.s.len() && self.s[self.pos] == b'.' {
            self.pos += 1;
            while self.pos < self.s.len() && self.s[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        if self.pos == start {
            bail!("expected a number at position {}", self.pos);
        }
        if self.pos < self.s.len() && (self.s[self.pos] | 0x20) == b'e' {
            let save = self.pos;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                self.pos += 1;
            }
            if self.pos < self.s.len() && self.s[self.pos].is_ascii_digit() {
                while self.pos < self.s.len() && self.s[self.pos].is_ascii_digit() {
                    self.pos += 1;
                }
            } else {
                self.pos = save;
            }
        }
        std::str::from_utf8(&self.s[start..self.pos])
            .ok()
            .and_then(|t| t.parse::<f64>().ok())
            .with_context(|| format!("invalid number at position {start}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            name: name.into(),
            arguments,
        }
    }

    #[test]
    fn calculator_arithmetic() {
        assert_eq!(eval_arithmetic("2+2*3").unwrap(), 8.0);
        assert_eq!(eval_arithmetic("(2+3)*4").unwrap(), 20.0);
        assert_eq!(eval_arithmetic("-2^2").unwrap(), -4.0);
        assert_eq!(eval_arithmetic("2^-3").unwrap(), 0.125);
        assert_eq!(eval_arithmetic("2 ^ 3 ^ 2").unwrap(), 512.0);
        assert_eq!(eval_arithmetic("10 / 4").unwrap(), 2.5);
        assert_eq!(eval_arithmetic("10 % 3").unwrap(), 1.0);
        assert_eq!(eval_arithmetic(" 1.5e2 + .5 ").unwrap(), 150.5);
        assert_eq!(eval_arithmetic("--4").unwrap(), 4.0);
        assert!(eval_arithmetic("1/0").is_err());
        assert!(eval_arithmetic("2+").is_err());
        assert!(eval_arithmetic("foo").is_err());
        assert_eq!(format_number(8.0), "8");
        assert_eq!(format_number(2.5), "2.5");
    }

    #[test]
    fn extract_plain_and_fenced_and_prose() {
        let plain =
            extract_tool_call(r#"{"name": "calculator", "arguments": {"expression": "2+2"}}"#)
                .unwrap();
        assert_eq!(plain.name, "calculator");
        assert_eq!(plain.arguments["expression"], "2+2");

        let fenced = extract_tool_call(
            "```json\n{\"name\": \"echo\", \"arguments\": {\"text\": \"hi\"}}\n```",
        )
        .unwrap();
        assert_eq!(fenced.name, "echo");
        assert_eq!(fenced.arguments["text"], "hi");

        let prose = extract_tool_call(
            "Sure! Here is the call:\n{\"name\": \"current_time\", \"arguments\": {}} thanks",
        )
        .unwrap();
        assert_eq!(prose.name, "current_time");
        assert_eq!(prose.arguments, serde_json::json!({}));

        let defaults = extract_tool_call(r#"{"name": "current_time"}"#).unwrap();
        assert_eq!(defaults.arguments, serde_json::json!({}));
    }

    #[test]
    fn extract_recovers_after_unclosed_fragment() {
        let text = "{\"name\": \"calculator\", \"arguments\": {\"expr";
        let full = r#" {"name": "echo", "arguments": {"text": "x"}}"#;
        let call = extract_tool_call(&format!("{text}{full}")).unwrap();
        assert_eq!(call.name, "echo");
        assert_eq!(call.arguments["text"], "x");
    }

    #[test]
    fn extract_rejects_garbage() {
        assert!(extract_tool_call("no json here").is_none());
        assert!(extract_tool_call("{\"name\": 42}").is_none());
        assert!(extract_tool_call("{\"arguments\": {}}").is_none());
        assert!(extract_tool_call("{\"name\": \"x\", \"arguments\": 3}").is_none());
    }

    #[test]
    fn extract_skips_nested_and_invalid_prefix() {
        let text = "{\"outer\": {\"name\": \"not-a-call\"}} {\"name\": \"echo\", \"arguments\": {\"text\": \"x\"}}";
        let call = extract_tool_call(text).unwrap();
        assert_eq!(call.name, "echo");
    }

    #[test]
    fn tool_execution_roundtrip() {
        assert!(execute(&call("echo", serde_json::json!({"text": "hey"}))).unwrap() == "hey");
        assert_eq!(
            execute(&call(
                "calculator",
                serde_json::json!({"expression": "6*7"})
            ))
            .unwrap(),
            "42"
        );
        let time = execute(&call("current_time", serde_json::json!({}))).unwrap();
        assert_eq!(time.len(), 23);
        assert!(time.ends_with(" UTC"));
        assert!(execute(&call("nope", serde_json::json!({}))).is_err());
    }

    #[test]
    fn extract_rejects_spec_echoes() {
        let spec = r#"{"description":"Returns the time.","name":"current_time","parameters":{"properties":{},"required":[],"type":"object"}}"#;
        let call = r#"{"name": "current_time", "arguments": {}}"#;
        assert!(
            extract_tool_call(spec).is_none(),
            "a tool spec is not a call"
        );
        let mixed = extract_tool_call(&format!("{spec}\n{call}")).unwrap();
        assert_eq!(mixed.name, "current_time");
        assert_eq!(mixed.arguments, serde_json::json!({}));
    }

    #[test]
    fn strip_tool_blocks_removes_echoes() {
        let text = "<tool>{\"name\":\"echo\"}</tool>\n{\"name\": \"echo\", \"arguments\": {\"text\": \"x\"}}";
        assert_eq!(
            strip_tool_blocks(text),
            "{\"name\": \"echo\", \"arguments\": {\"text\": \"x\"}}"
        );
        assert_eq!(strip_tool_blocks("<tool>unclosed"), "");
        assert_eq!(strip_tool_blocks("plain answer"), "plain answer");
    }

    #[test]
    fn system_prompt_lists_tools() {
        let block = system_prompt(&["calculator"]);
        assert!(block.contains("<tool>"));
        assert!(block.contains(r#""name":"calculator""#));
        assert!(block.contains("ONLY one JSON object"));
        let block = system_prompt(&["nope"]);
        assert!(!block.contains("<tool>{"));
    }
}
