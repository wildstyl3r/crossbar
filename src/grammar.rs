//! Schema-constrained sampling: a vocabulary token trie plus a char-driven
//! JSON state machine mask logits so sampling can only produce text that
//! stays consistent with a (subset of) JSON Schema — used for structured
//! output and forced tool-call envelopes.
//!
//! Supported schema subset: objects with declared properties (required
//! first, in declaration order; each key at most once), arrays, strings,
//! numbers/integers, booleans, null, and string enums/consts. `type`
//! arrays, regex patterns, nested free-form objects (schema-less `{}`
//! values) and non-string enums are not supported.

use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::generation::LogitsFilter;

const MASKED: f32 = -1e30;

fn is_ws(ch: u8) -> bool {
    matches!(ch, b' ' | b'\t' | b'\n' | b'\r')
}

fn is_number_char(ch: u8) -> bool {
    ch.is_ascii_digit() || matches!(ch, b'.' | b'e' | b'E' | b'+' | b'-')
}

fn is_str_char(ch: u8) -> bool {
    ch >= 0x20 && ch != b'"' && ch != b'\\'
}

#[derive(Debug)]
enum Schema {
    /// Any scalar value (string / number / bool / null); objects and
    /// arrays under a schema-less value are not supported.
    Any,
    Object(Arc<ObjSpec>),
    Array(Arc<Schema>),
    Str,
    Number,
    Bool,
    Null,
    EnumStr(Arc<[String]>),
}

#[derive(Debug)]
struct ObjSpec {
    /// All allowed keys: required first, then optional (declaration order).
    keys: Vec<String>,
    required_count: usize,
    /// Property schemas, parallel to `keys`.
    props: Vec<Arc<Schema>>,
}

/// A schema *definition* document, deserialized by serde. Unknown fields
/// (descriptions, format, …) are ignored.
#[derive(Debug, serde::Deserialize)]
struct SchemaDef {
    #[serde(rename = "type")]
    ty: Option<TypeSpec>,
    properties: Option<std::collections::BTreeMap<String, SchemaDef>>,
    required: Option<Vec<String>>,
    #[serde(rename = "enum")]
    enum_values: Option<Vec<Value>>,
    #[serde(rename = "const")]
    const_value: Option<Value>,
    items: Option<Box<SchemaDef>>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(untagged)]
enum TypeSpec {
    Name(String),
    List(Vec<String>),
}

fn compile_schema(v: &Value) -> Result<Arc<Schema>> {
    let def: SchemaDef = serde_json::from_value(v.clone()).context("invalid schema")?;
    Ok(Arc::new(build_schema(def)?))
}

/// Validate that a schema only uses the supported subset.
pub fn validate_schema(schema: &Value) -> Result<()> {
    compile_schema(schema).map(|_| ())
}

fn build_schema(def: SchemaDef) -> Result<Schema> {
    if let Some(c) = def.const_value {
        return match c.as_str() {
            Some(s) => Ok(Schema::EnumStr(vec![s.to_string()].into())),
            None => bail!("only string \"const\" values are supported"),
        };
    }
    if let Some(list) = def.enum_values {
        let mut options = Vec::with_capacity(list.len());
        for item in list {
            match item.as_str() {
                Some(s) => options.push(s.to_string()),
                None => bail!("only string enums are supported"),
            }
        }
        return Ok(Schema::EnumStr(options.into()));
    }
    match def.ty {
        Some(TypeSpec::List(types)) => {
            bail!("\"type\" arrays are not supported (got {types:?})")
        }
        Some(TypeSpec::Name(name)) => match name.as_str() {
            "object" => {
                let properties = def
                    .properties
                    .context("object schema needs a \"properties\" object")?;
                build_object(properties, def.required)
            }
            "array" => Ok(Schema::Array(match def.items {
                Some(items) => Arc::new(build_schema(*items)?),
                None => Arc::new(Schema::Any),
            })),
            "string" => Ok(Schema::Str),
            "number" | "integer" => Ok(Schema::Number),
            "boolean" => Ok(Schema::Bool),
            "null" => Ok(Schema::Null),
            other => bail!("unsupported schema type {other:?}"),
        },
        None => match def.properties {
            Some(properties) => build_object(properties, def.required),
            None => Ok(Schema::Any),
        },
    }
}

fn build_object(
    properties: std::collections::BTreeMap<String, SchemaDef>,
    required: Option<Vec<String>>,
) -> Result<Schema> {
    let required = required.unwrap_or_default();
    let mut props: Vec<(String, Arc<Schema>)> = Vec::with_capacity(properties.len());
    for (name, def) in properties {
        props.push((name, Arc::new(build_schema(def)?)));
    }
    let mut keys: Vec<String> = Vec::new();
    for name in &required {
        if props.iter().any(|(p, _)| p == name) {
            keys.push((*name).to_string());
        }
    }
    for (name, _) in &props {
        if !required.contains(name) {
            keys.push(name.clone());
        }
    }
    let required_count = required
        .iter()
        .filter(|r| keys.iter().any(|k| k == *r))
        .count();
    let props = keys
        .iter()
        .map(|k| {
            props
                .iter()
                .find(|(p, _)| p == k)
                .map(|(_, s)| s.clone())
                .expect("key was built from props")
        })
        .collect();
    Ok(Schema::Object(Arc::new(ObjSpec {
        keys,
        required_count,
        props,
    })))
}

fn schema_accepts_value(schema: &Schema, ch: u8) -> bool {
    match schema {
        Schema::Any => matches!(ch, b'"' | b't' | b'f' | b'n') || ch.is_ascii_digit() || ch == b'-',
        Schema::Object(_) => ch == b'{',
        Schema::Array(_) => ch == b'[',
        Schema::Str | Schema::EnumStr(_) => ch == b'"',
        Schema::Number => ch.is_ascii_digit() || ch == b'-',
        Schema::Bool => ch == b't' || ch == b'f',
        Schema::Null => ch == b'n',
    }
}

/// The concrete frame for a value whose first char `ch` was just consumed.
fn dispatch_value(schema: &Schema, ch: u8) -> Frame {
    match schema {
        Schema::Object(spec) => Frame::Object {
            spec: spec.clone(),
            remaining: (0..spec.keys.len()).collect(),
            required_left: spec.required_count,
            last_key: 0,
            phase: ObjPhase::ExpectKeyOrEnd,
        },
        Schema::Array(items) => Frame::Array {
            items: items.clone(),
            phase: ArrPhase::ExpectValueOrEnd,
        },
        Schema::Str => Frame::Str,
        Schema::Number => Frame::Number,
        Schema::Bool => Frame::Literal {
            text: if ch == b't' { "true" } else { "false" },
            pos: 1,
        },
        Schema::Null => Frame::Literal {
            text: "null",
            pos: 1,
        },
        Schema::EnumStr(options) => Frame::EnumStr {
            options: options.clone(),
            buf: Vec::new(),
        },
        Schema::Any => match ch {
            b'"' => Frame::Str,
            b't' => Frame::Literal {
                text: "true",
                pos: 1,
            },
            b'f' => Frame::Literal {
                text: "false",
                pos: 1,
            },
            b'n' => Frame::Literal {
                text: "null",
                pos: 1,
            },
            _ => Frame::Number,
        },
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ObjPhase {
    /// Fresh object: a key or the closing brace (no trailing commas).
    ExpectKeyOrEnd,
    KeyPending,
    ExpectColon,
    ValuePending,
    /// After a value: a comma (then a key) or the closing brace.
    ExpectCommaOrEnd,
    /// After a comma: only a key.
    ExpectKey,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ArrPhase {
    ExpectValueOrEnd,
    ValuePending,
    ExpectCommaOrEnd,
}

#[derive(Clone)]
enum Frame {
    /// A value start: dispatched (and replaced) by the first value char.
    Value(Arc<Schema>),
    Object {
        spec: Arc<ObjSpec>,
        remaining: Vec<usize>,
        required_left: usize,
        last_key: usize,
        phase: ObjPhase,
    },
    Array {
        items: Arc<Schema>,
        phase: ArrPhase,
    },
    Str,
    StrEscape,
    EnumStr {
        options: Arc<[String]>,
        buf: Vec<u8>,
    },
    Number,
    Literal {
        text: &'static str,
        pos: usize,
    },
    Done,
}

#[derive(Clone)]
struct Machine {
    stack: Vec<Frame>,
    /// Consecutive whitespace chars consumed anywhere; bounded so the
    /// model cannot loop forever on whitespace between structure.
    ws_run: usize,
}

/// Whitespace runs longer than this are refused (pretty-printing needs
/// only a few; degenerate repetition needs unbounded).
const WS_LIMIT: usize = 8;

enum Step {
    Consumed,
    /// Consumed and this frame completed (pop and notify the parent).
    ConsumedDone,
    /// Not consumed; this frame completed so the parent may consume.
    PassDown,
    Rejected,
}

impl Machine {
    fn new(top: Arc<Schema>) -> Self {
        Self {
            stack: vec![Frame::Value(top)],
            ws_run: 0,
        }
    }

    fn is_done(&self) -> bool {
        matches!(self.stack.last(), Some(Frame::Done))
    }

    fn ws_ok(&self, ch: u8) -> bool {
        !is_ws(ch) || self.ws_run < WS_LIMIT
    }

    fn bump_ws(&mut self, ch: u8) {
        if is_ws(ch) {
            self.ws_run += 1;
        } else {
            self.ws_run = 0;
        }
    }

    fn step_top(&mut self, ch: u8) -> Step {
        let mut push = None;
        let ws_ok = self.ws_ok(ch);
        let step = {
            let frame = self
                .stack
                .last_mut()
                .expect("machine stack must not be empty");
            match frame {
                Frame::Value(schema) => {
                    if ws_ok && is_ws(ch) {
                        Step::Consumed
                    } else if !schema_accepts_value(schema, ch) {
                        Step::Rejected
                    } else {
                        *frame = dispatch_value(schema, ch);
                        Step::Consumed
                    }
                }
                Frame::Object {
                    spec,
                    remaining,
                    required_left,
                    last_key,
                    phase,
                } => match phase {
                    ObjPhase::ExpectKeyOrEnd => {
                        if ws_ok && is_ws(ch) {
                            Step::Consumed
                        } else if ch == b'"' && !remaining.is_empty() {
                            let options: Vec<String> =
                                remaining.iter().map(|&i| spec.keys[i].clone()).collect();
                            *phase = ObjPhase::KeyPending;
                            push = Some(Frame::EnumStr {
                                options: options.into(),
                                buf: Vec::new(),
                            });
                            Step::Consumed
                        } else if ch == b'}' && *required_left == 0 {
                            Step::ConsumedDone
                        } else {
                            Step::Rejected
                        }
                    }
                    ObjPhase::ExpectKey => {
                        if ws_ok && is_ws(ch) {
                            Step::Consumed
                        } else if ch == b'"' && !remaining.is_empty() {
                            let options: Vec<String> =
                                remaining.iter().map(|&i| spec.keys[i].clone()).collect();
                            *phase = ObjPhase::KeyPending;
                            push = Some(Frame::EnumStr {
                                options: options.into(),
                                buf: Vec::new(),
                            });
                            Step::Consumed
                        } else {
                            Step::Rejected
                        }
                    }
                    ObjPhase::ExpectColon => {
                        if ws_ok && is_ws(ch) {
                            Step::Consumed
                        } else if ch == b':' {
                            *phase = ObjPhase::ValuePending;
                            push = Some(Frame::Value(spec.props[*last_key].clone()));
                            Step::Consumed
                        } else {
                            Step::Rejected
                        }
                    }
                    ObjPhase::ExpectCommaOrEnd => {
                        if ws_ok && is_ws(ch) {
                            Step::Consumed
                        } else if ch == b',' {
                            *phase = ObjPhase::ExpectKey;
                            Step::Consumed
                        } else if ch == b'}' && *required_left == 0 {
                            Step::ConsumedDone
                        } else {
                            Step::Rejected
                        }
                    }
                    _ => Step::Rejected,
                },
                Frame::Array { items, phase } => match phase {
                    ArrPhase::ExpectValueOrEnd => {
                        if ws_ok && is_ws(ch) {
                            Step::Consumed
                        } else if ch == b']' {
                            Step::ConsumedDone
                        } else if schema_accepts_value(items, ch) {
                            *phase = ArrPhase::ValuePending;
                            push = Some(dispatch_value(items, ch));
                            Step::Consumed
                        } else {
                            Step::Rejected
                        }
                    }
                    ArrPhase::ExpectCommaOrEnd => {
                        if ws_ok && is_ws(ch) {
                            Step::Consumed
                        } else if ch == b',' {
                            *phase = ArrPhase::ExpectValueOrEnd;
                            Step::Consumed
                        } else if ch == b']' {
                            Step::ConsumedDone
                        } else {
                            Step::Rejected
                        }
                    }
                    _ => Step::Rejected,
                },
                Frame::Str => {
                    if ch == b'"' {
                        Step::ConsumedDone
                    } else if ch == b'\\' {
                        push = Some(Frame::StrEscape);
                        Step::Consumed
                    } else if ch >= 0x20 {
                        Step::Consumed
                    } else {
                        Step::Rejected
                    }
                }
                Frame::StrEscape => Step::ConsumedDone,
                Frame::EnumStr { options, buf } => {
                    if ch == b'"' {
                        if options.iter().any(|o| o.as_bytes() == buf) {
                            Step::ConsumedDone
                        } else {
                            Step::Rejected
                        }
                    } else if ch >= 0x20
                        && ch != b'\\'
                        && options
                            .iter()
                            .any(|o| o.as_bytes().get(buf.len()) == Some(&ch))
                    {
                        buf.push(ch);
                        Step::Consumed
                    } else {
                        Step::Rejected
                    }
                }
                Frame::Number => {
                    if is_number_char(ch) {
                        Step::Consumed
                    } else {
                        Step::PassDown
                    }
                }
                Frame::Literal { text, pos } => {
                    if text.as_bytes().get(*pos) == Some(&ch) {
                        if *pos + 1 == text.len() {
                            Step::ConsumedDone
                        } else {
                            *pos += 1;
                            Step::Consumed
                        }
                    } else {
                        Step::Rejected
                    }
                }
                Frame::Done => Step::Rejected,
            }
        };
        if let Some(frame) = push {
            self.stack.push(frame);
        }
        step
    }

    fn complete_top(&mut self) {
        let done = self.stack.pop();
        match (self.stack.last_mut(), done) {
            (
                Some(Frame::Object {
                    spec,
                    remaining,
                    required_left,
                    last_key,
                    phase,
                }),
                Some(Frame::EnumStr { buf, .. }),
            ) if *phase == ObjPhase::KeyPending => {
                if let Some(pos) = remaining
                    .iter()
                    .position(|&i| spec.keys[i].as_bytes() == buf)
                {
                    let ki = remaining.remove(pos);
                    *last_key = ki;
                    if ki < spec.required_count {
                        *required_left -= 1;
                    }
                }
                *phase = ObjPhase::ExpectColon;
            }
            (Some(Frame::Object { phase, .. }), _) => {
                *phase = ObjPhase::ExpectCommaOrEnd;
            }
            (Some(Frame::Array { phase, .. }), _) => {
                *phase = ArrPhase::ExpectCommaOrEnd;
            }
            (Some(Frame::Value(_)), _) => unreachable!("value frames are replaced, never done"),
            _ => {
                self.stack.push(Frame::Done);
            }
        }
    }

    /// Feed one char; `false` on rejection (the machine is then in an
    /// unspecified state and must be reset by the caller).
    fn feed(&mut self, ch: u8) -> bool {
        for _ in 0..256 {
            if self.stack.is_empty() {
                self.stack.push(Frame::Done);
                return false;
            }
            match self.step_top(ch) {
                Step::Consumed => {
                    self.bump_ws(ch);
                    return true;
                }
                Step::ConsumedDone => {
                    self.complete_top();
                    self.bump_ws(ch);
                    return true;
                }
                Step::PassDown => {
                    self.complete_top();
                }
                Step::Rejected => return false,
            }
        }
        false
    }

    /// Whether `ch` would be accepted right now (must agree with `feed`).
    fn accepts(&self, ch: u8) -> bool {
        match self.stack.last() {
            Some(Frame::Number) if !is_number_char(ch) => {
                // the number frame pops and the parent sees the char
                if self.stack.len() < 2 {
                    return false;
                }
                match &self.stack[self.stack.len() - 2] {
                    Frame::Object { required_left, .. } => {
                        (is_ws(ch) && self.ws_ok(ch))
                            || ch == b','
                            || (ch == b'}' && *required_left == 0)
                    }
                    Frame::Array { .. } => {
                        (is_ws(ch) && self.ws_ok(ch)) || ch == b',' || ch == b']'
                    }
                    _ => false,
                }
            }
            Some(frame) => matches!(frame_accepts(frame, ch, self.ws_run), Verdict::Accept),
            None => false,
        }
    }

    /// Whether consuming `ch` changes the machine state (used by the trie
    /// DFS to skip cloning on pure-continuation chars).
    fn changes_state(&self, ch: u8) -> bool {
        match self.stack.last() {
            Some(Frame::Str) => !is_str_char(ch),
            Some(Frame::Number) => !is_number_char(ch),
            _ => true,
        }
    }
}

enum Verdict {
    Accept,
    /// The frame completes without consuming; the parent sees the char.
    PassDown,
    Reject,
}

/// Pure per-frame char check, mirroring `step_top` (must agree with it).
fn frame_accepts(frame: &Frame, ch: u8, ws_run: usize) -> Verdict {
    match frame {
        Frame::Value(schema) => {
            if (is_ws(ch) && ws_run < WS_LIMIT) || schema_accepts_value(schema, ch) {
                Verdict::Accept
            } else {
                Verdict::Reject
            }
        }
        Frame::Object {
            remaining,
            required_left,
            phase,
            ..
        } => match phase {
            ObjPhase::ExpectKeyOrEnd => {
                if (is_ws(ch) && ws_run < WS_LIMIT)
                    || (ch == b'"' && !remaining.is_empty())
                    || (ch == b'}' && *required_left == 0)
                {
                    Verdict::Accept
                } else {
                    Verdict::Reject
                }
            }
            ObjPhase::ExpectKey => {
                if (is_ws(ch) && ws_run < WS_LIMIT) || (ch == b'"' && !remaining.is_empty()) {
                    Verdict::Accept
                } else {
                    Verdict::Reject
                }
            }
            ObjPhase::ExpectColon => {
                if (is_ws(ch) && ws_run < WS_LIMIT) || ch == b':' {
                    Verdict::Accept
                } else {
                    Verdict::Reject
                }
            }
            ObjPhase::ExpectCommaOrEnd => {
                if (is_ws(ch) && ws_run < WS_LIMIT)
                    || ch == b','
                    || (ch == b'}' && *required_left == 0)
                {
                    Verdict::Accept
                } else {
                    Verdict::Reject
                }
            }
            _ => Verdict::Reject,
        },
        Frame::Array { items, phase } => match phase {
            ArrPhase::ExpectValueOrEnd => {
                if (is_ws(ch) && ws_run < WS_LIMIT) || ch == b']' || schema_accepts_value(items, ch)
                {
                    Verdict::Accept
                } else {
                    Verdict::Reject
                }
            }
            ArrPhase::ExpectCommaOrEnd => {
                if (is_ws(ch) && ws_run < WS_LIMIT) || ch == b',' || ch == b']' {
                    Verdict::Accept
                } else {
                    Verdict::Reject
                }
            }
            _ => Verdict::Reject,
        },
        Frame::Str => {
            if ch == b'"' || is_str_char(ch) {
                Verdict::Accept
            } else {
                Verdict::Reject
            }
        }
        Frame::StrEscape => Verdict::Accept,
        Frame::EnumStr { options, buf } => {
            if (ch == b'"' && options.iter().any(|o| o.as_bytes() == buf))
                || (is_str_char(ch)
                    && options
                        .iter()
                        .any(|o| o.as_bytes().get(buf.len()) == Some(&ch)))
            {
                Verdict::Accept
            } else {
                Verdict::Reject
            }
        }
        Frame::Number => {
            if is_number_char(ch) {
                Verdict::Accept
            } else {
                Verdict::PassDown
            }
        }
        Frame::Literal { text, pos } => {
            if text.as_bytes().get(*pos) == Some(&ch) {
                Verdict::Accept
            } else {
                Verdict::Reject
            }
        }
        Frame::Done => Verdict::Reject,
    }
}

struct Node {
    children: Vec<(u8, u32)>,
    terminal: Option<u32>,
}

struct TokenTrie {
    nodes: Vec<Node>,
    str_offsets: Vec<u32>,
    str_ids: Vec<u32>,
    close_offsets: Vec<u32>,
    close_ids: Vec<u32>,
}

impl TokenTrie {
    fn new() -> Self {
        Self {
            nodes: vec![Node {
                children: Vec::new(),
                terminal: None,
            }],
            str_offsets: Vec::new(),
            str_ids: Vec::new(),
            close_offsets: Vec::new(),
            close_ids: Vec::new(),
        }
    }

    fn insert(&mut self, id: u32, text: &str) {
        let mut node = 0usize;
        for &b in text.as_bytes() {
            node = match self.nodes[node].children.iter().find(|(c, _)| *c == b) {
                Some(&(_, n)) => n as usize,
                None => {
                    let n = self.nodes.len();
                    self.nodes.push(Node {
                        children: Vec::new(),
                        terminal: None,
                    });
                    self.nodes[node].children.push((b, n as u32));
                    n
                }
            };
        }
        if self.nodes[node].terminal.is_none() {
            self.nodes[node].terminal = Some(id);
        }
    }

    /// Per node, the ids of tokens whose continuation from that node is
    /// plain JSON-string content (no quotes, backslashes or control
    /// chars), terminating anywhere. Tokens that close the string or open
    /// an escape are rare by comparison and handled by the DFS recursion.
    fn build_str_ids(&mut self) {
        let n = self.nodes.len();
        let mut lists: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
        for i in (0..n).rev() {
            let mut list = Vec::new();
            if let Some(id) = self.nodes[i].terminal {
                list.push(id);
            }
            for &(ch, child) in &self.nodes[i].children {
                if is_str_char(ch) {
                    list.extend_from_slice(&lists[child as usize]);
                }
            }
            lists[i] = list;
        }
        (self.str_offsets, self.str_ids) = flatten(lists);
    }

    /// Per node, ids of tokens that continue with plain string chars and
    /// end exactly on a closing quote or a backslash (string continues in
    /// the next token via the escape state).
    fn build_close_ids(&mut self) {
        let n = self.nodes.len();
        let mut lists: Vec<Vec<u32>> = (0..n).map(|_| Vec::new()).collect();
        for i in (0..n).rev() {
            let mut list = Vec::new();
            for &(ch, child) in &self.nodes[i].children {
                if ch == b'"' || ch == b'\\' {
                    if let Some(id) = self.nodes[child as usize].terminal {
                        list.push(id);
                    }
                } else if is_str_char(ch) {
                    list.extend_from_slice(&lists[child as usize]);
                }
            }
            lists[i] = list;
        }
        (self.close_offsets, self.close_ids) = flatten(lists);
    }

    fn str_ids(&self, node: usize) -> &[u32] {
        &self.str_ids[self.str_offsets[node] as usize..self.str_offsets[node + 1] as usize]
    }

    fn close_ids(&self, node: usize) -> &[u32] {
        &self.close_ids[self.close_offsets[node] as usize..self.close_offsets[node + 1] as usize]
    }
}

fn flatten(lists: Vec<Vec<u32>>) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = Vec::with_capacity(lists.len() + 1);
    let mut flat = Vec::new();
    for list in &lists {
        offsets.push(flat.len() as u32);
        flat.extend_from_slice(list);
    }
    offsets.push(flat.len() as u32);
    (offsets, flat)
}

pub struct SchemaMasker {
    trie: TokenTrie,
    machine: Machine,
    initial: Machine,
    eos: Vec<u32>,
    token_bytes: Vec<Option<Box<[u8]>>>,
}

impl SchemaMasker {
    /// Build the masker from the tokenizer: decodes every vocab entry once
    /// (special tokens and undecodable byte pieces are excluded) and inserts
    /// the strings into the trie. Startup cost is a few seconds for a
    /// 262k-entry vocabulary.
    pub fn new(tokenizer: &Tokenizer, schema: &Value, eos_tokens: Vec<u32>) -> Result<Self> {
        let vocab = tokenizer.get_vocab_size(true);
        let mut pairs = Vec::with_capacity(vocab);
        for id in 0..vocab {
            let Ok(text) = tokenizer.decode(&[id as u32], true) else {
                continue;
            };
            if text.is_empty() || text.contains('\u{FFFD}') {
                continue;
            }
            pairs.push((id as u32, text));
        }
        Ok(Self::from_parts(schema, eos_tokens, pairs))
    }

    pub fn from_parts(
        schema: &Value,
        eos_tokens: Vec<u32>,
        pairs: impl IntoIterator<Item = (u32, String)>,
    ) -> Self {
        let top = compile_schema(schema).expect("schema must compile");
        let mut trie = TokenTrie::new();
        let vocab = pairs
            .into_iter()
            .inspect(|(id, text)| {
                trie.insert(*id, text);
            })
            .fold(0usize, |acc, (id, _)| acc.max(id as usize + 1))
            .max(1);
        trie.build_str_ids();
        trie.build_close_ids();
        let mut token_bytes: Vec<Option<Box<[u8]>>> = vec![None; vocab];
        // rebuild token bytes from the trie paths
        let mut stack = vec![(0usize, Vec::<u8>::new())];
        while let Some((node, mut path)) = stack.pop() {
            if let Some(id) = trie.nodes[node].terminal {
                token_bytes[id as usize] = Some(path.clone().into());
            }
            for &(ch, child) in &trie.nodes[node].children {
                path.push(ch);
                stack.push((child as usize, path.clone()));
                path.pop();
            }
        }
        let machine = Machine::new(top);
        Self {
            trie,
            initial: machine.clone(),
            machine,
            eos: eos_tokens,
            token_bytes,
        }
    }

    /// All vocab ids whose next token keeps the output schema-valid
    /// (`None` means no constraint could be derived; do not mask).
    fn allowed(&self) -> Option<Vec<u32>> {
        if self.machine.is_done() {
            return (!self.eos.is_empty()).then(|| self.eos.clone());
        }
        let mut out = Vec::new();
        self.dfs(0, &self.machine, &mut out);
        (!out.is_empty()).then_some(out)
    }

    fn dfs(&self, node: usize, state: &Machine, out: &mut Vec<u32>) {
        match state.stack.last() {
            Some(Frame::Str) => {
                out.extend_from_slice(self.trie.str_ids(node));
                out.extend_from_slice(self.trie.close_ids(node));
                for &(ch, child) in &self.trie.nodes[node].children {
                    if ch == b'"' || ch == b'\\' {
                        let mut next = state.clone();
                        let fed = next.feed(ch);
                        debug_assert!(fed);
                        self.dfs(child as usize, &next, out);
                    }
                }
                return;
            }
            Some(Frame::StrEscape) => {
                if let Some(id) = self.trie.nodes[node].terminal {
                    out.push(id);
                }
                for &(ch, child) in &self.trie.nodes[node].children {
                    let mut next = state.clone();
                    let fed = next.feed(ch);
                    debug_assert!(fed);
                    self.dfs(child as usize, &next, out);
                }
                return;
            }
            _ => {}
        }
        if let Some(id) = self.trie.nodes[node].terminal {
            out.push(id);
        }
        for &(ch, child) in &self.trie.nodes[node].children {
            if !state.accepts(ch) {
                continue;
            }
            if state.changes_state(ch) {
                let mut next = state.clone();
                let fed = next.feed(ch);
                debug_assert!(fed);
                self.dfs(child as usize, &next, out);
            } else {
                self.dfs(child as usize, state, out);
            }
        }
    }
}

impl LogitsFilter for SchemaMasker {
    fn mask(&mut self, logits: &mut [f32]) {
        let Some(allowed) = self.allowed() else {
            eprintln!("[grammar] no allowed set; not masking");
            return;
        };
        let mut keep = vec![false; logits.len()];
        for id in allowed {
            let i = id as usize;
            if i < logits.len() {
                keep[i] = true;
            }
        }
        for (logit, keep) in logits.iter_mut().zip(keep) {
            if !keep {
                *logit = MASKED;
            }
        }
    }

    fn on_token(&mut self, token: u32) {
        let Some(Some(bytes)) = self.token_bytes.get(token as usize) else {
            return;
        };
        for &b in bytes.iter() {
            if !self.machine.feed(b) {
                eprintln!("[grammar] constraint desync at token {token}; resetting");
                self.machine = self.initial.clone();
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn machine_for(schema: Value) -> Machine {
        Machine::new(compile_schema(&schema).unwrap())
    }
    fn machine_for_borrowed(schema: &Value) -> Machine {
        Machine::new(compile_schema(schema).unwrap())
    }

    fn feed_str(m: &mut Machine, s: &str) -> bool {
        s.bytes().all(|b| m.feed(b))
    }

    #[test]
    fn object_with_required_keys() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "value": { "type": "number" },
            },
            "required": ["name", "value"],
        });
        let mut m = machine_for(schema);
        assert!(feed_str(&mut m, r#"{"name":"x","value":42}"#));
        assert!(m.is_done());
    }

    #[test]
    fn required_keys_must_appear_before_close() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": { "type": "string" },
                "b": { "type": "number" },
            },
            "required": ["a"],
        });
        let mut m = machine_for_borrowed(&schema);
        assert!(feed_str(&mut m, r#"{"b":1,"a":"x"}"#));
        assert!(m.is_done());

        let mut m = machine_for_borrowed(&schema);
        assert!(feed_str(&mut m, r#"{"b":1"#));
        assert!(
            !m.feed(b'}'),
            "closing with a required key missing must fail"
        );
    }

    #[test]
    fn optional_keys_close_immediately() {
        let schema = json!({
            "type": "object",
            "properties": { "a": { "type": "number" } },
            "required": [],
        });
        let mut m = machine_for(schema);
        assert!(feed_str(&mut m, "{}"));
        assert!(m.is_done());
    }

    #[test]
    fn full_document_feed_reports_first_rejected_char() {
        let schema = json!({
            "type": "object",
            "properties": {
                "items": { "type": "array", "items": {} },
                "flag": { "type": "boolean" },
                "nothing": { "type": "null" },
            },
            "required": ["items", "flag", "nothing"],
        });
        let doc =
            r#"{ "items": [ 1, 2.5e3, -7, true, null, "s" ], "flag": true, "nothing": null }"#;
        let mut m = machine_for_borrowed(&schema);
        for (i, b) in doc.bytes().enumerate() {
            if !m.feed(b) {
                panic!(
                    "rejected {:?} at position {i} (prefix {:?})",
                    b as char,
                    &doc[..=i]
                );
            }
        }
    }

    #[test]
    fn arrays_scalars_literals_and_whitespace() {
        let schema = json!({
            "type": "object",
            "properties": {
                "items": { "type": "array", "items": {} },
                "flag": { "type": "boolean" },
                "nothing": { "type": "null" },
            },
            "required": ["items", "flag", "nothing"],
        });
        let mut m = machine_for(schema);
        let doc =
            r#"{ "items": [ 1, 2.5e3, -7, true, null, "s" ], "flag": true, "nothing": null }"#;
        assert!(
            feed_str(&mut m, doc),
            "full valid document must be accepted"
        );
        assert!(m.is_done());
    }

    #[test]
    fn string_enums() {
        let schema = json!({
            "type": "object",
            "properties": { "mode": { "enum": ["a", "ab"] } },
            "required": ["mode"],
        });
        let mut m = machine_for_borrowed(&schema);
        assert!(feed_str(&mut m, r#"{"mode":"ab"}"#));
        assert!(m.is_done());

        let mut m = machine_for_borrowed(&schema);
        assert!(feed_str(&mut m, r#"{"mode":"a"#));
        assert!(!m.feed(b'x'), "x diverges from every option");
    }

    #[test]
    fn rejects_malformed_documents() {
        let schema = json!({
            "type": "object",
            "properties": { "a": { "type": "number" } },
            "required": ["a"],
        });
        let mut m = machine_for_borrowed(&schema);
        assert!(!feed_str(&mut m, r#"{"a":1,}"#), "trailing comma rejected");
        let mut m = machine_for_borrowed(&schema);
        assert!(
            !feed_str(&mut m, r#"{"a":"str"}"#),
            "wrong value type rejected"
        );
    }

    #[test]
    fn accepts_agrees_with_feed() {
        let schema = json!({
            "type": "object",
            "properties": {
                "k": { "type": "string" },
                "n": { "type": "number" },
                "m": { "enum": ["ab", "abc"] },
                "b": { "type": "boolean" },
            },
            "required": ["k"],
        });
        let prefixes = [
            "",
            "{",
            "{\"",
            "{\"k",
            "{\"k\"",
            "{\"k\":",
            "{\"k\":\"v\\",
            "{\"k\":\"v\",\"",
            "{\"k\":\"v\",\"n",
            "{\"k\":\"v\",\"n\":1",
            "{\"k\":\"v\",\"n\":1.",
            "{\"k\":\"v\",\"n\":1,\"",
            "{\"k\":\"v\",\"n\":1,\"m\":\"a",
            "{\"k\":\"v\",\"n\":1,\"m\":\"ab\"",
            "{\"k\":\"v\",\"n\":1,\"m\":\"ab\",\"",
            "{\"k\":\"v\",\"n\":1,\"m\":\"ab\",\"b\":tr",
        ];
        for prefix in prefixes {
            for ch in 0..=255u8 {
                let mut m = machine_for_borrowed(&schema);
                assert!(feed_str(&mut m, prefix), "prefix {prefix:?} must be valid");
                let accepts = m.accepts(ch);
                let mut probe = m.clone();
                assert_eq!(
                    accepts,
                    probe.feed(ch),
                    "accepts/feed mismatch at prefix {prefix:?}, ch {ch}"
                );
            }
        }
    }

    #[test]
    fn unsupported_schemas_are_rejected() {
        assert!(compile_schema(&json!({"type": ["string", "null"]})).is_err());
        assert!(compile_schema(&json!({"enum": [1, 2]})).is_err());
        assert!(compile_schema(&json!({"type": "widget"})).is_err());
        assert!(compile_schema(&json!("just a string")).is_err());
    }

    #[test]
    fn masker_drives_greedy_generation_to_valid_json() {
        let schema = json!({
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"],
        });
        // synthetic vocab: eos=0, then the pieces of {"name": "x"}
        let pairs = [
            (1, "{".to_string()),
            (2, "\"".to_string()),
            (3, "name".to_string()),
            (4, ":".to_string()),
            (5, "x\"".to_string()),
            (6, "\"}".to_string()),
            (7, " ".to_string()),
            (8, ",".to_string()),
            (9, "}".to_string()),
        ];
        let texts = ["{", "\"", "name", ":", "x\"", "\"}", " ", ",", "}"];
        let expected = r#"{"name": "x"}"#;
        let mut m = SchemaMasker::from_parts(&schema, vec![0], pairs);
        let mut text = String::new();
        for _ in 0..20 {
            let allowed = m.allowed().expect("some token must be allowed");
            if allowed.contains(&0) {
                break;
            }
            let token = allowed
                .iter()
                .copied()
                .find(|&t| expected[text.len()..].starts_with(texts[(t - 1) as usize]))
                .unwrap_or_else(|| panic!("the canonical continuation must be allowed; allowed = {allowed:?}, at {:?}", &expected[text.len()..]));
            m.on_token(token);
            text.push_str(texts[(token - 1) as usize]);
        }
        assert_eq!(text, expected);
    }

    #[test]
    fn masker_masks_invalid_tokens() {
        let schema = json!({
            "type": "object",
            "properties": { "a": { "type": "number" } },
            "required": ["a"],
        });
        let pairs = [
            (1, "{".to_string()),
            (2, "\"".to_string()),
            (3, "a".to_string()),
            (4, "\":".to_string()),
            (5, "1".to_string()),
            (6, "\"".to_string()),
            (7, "}".to_string()),
            (8, "x".to_string()),
        ];
        let mut m = SchemaMasker::from_parts(&schema, vec![0], pairs);
        let allowed = m.allowed().unwrap();
        assert!(allowed.contains(&1));
        assert!(
            !allowed.contains(&8),
            "bare text cannot start a JSON object"
        );

        m.on_token(1); // "{"
        let allowed = m.allowed().unwrap();
        assert!(allowed.contains(&2));
        assert!(!allowed.contains(&7), "the required key is still missing");
        assert!(!allowed.contains(&3), "keys must start with a quote");

        m.on_token(2);
        m.on_token(3);
        let allowed = m.allowed().unwrap();
        assert!(allowed.contains(&4), "close the key then colon");
        assert!(!allowed.contains(&5));
    }
}
