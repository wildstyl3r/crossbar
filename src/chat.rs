use anyhow::{Context, Result};
use minijinja::{Environment, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
    /// A tool result, rendered as a user turn (Gemma has no tool role).
    ToolResult(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
        }
    }

    pub fn tool_result(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::ToolResult(name.into()),
            content: content.into(),
        }
    }
}

pub fn render(template: &str, system: Option<&str>, prompt: &str) -> Result<String> {
    let mut messages = Vec::new();
    if let Some(system) = system {
        messages.push(Message::system(system));
    }
    messages.push(Message::user(prompt));
    render_messages(template, &messages)
}

/// Render a full conversation through the chat template. The system message
/// (first only) is folded into the first user turn; tool results become
/// framed user turns so the strict user/assistant alternation of the Gemma
/// template holds.
pub fn render_messages(template: &str, messages: &[Message]) -> Result<String> {
    anyhow::ensure!(!messages.is_empty(), "cannot render an empty conversation");
    let mut system_prefix: Option<String> = None;
    let mut rendered: Vec<serde_json::Value> = Vec::new();
    for message in messages {
        match &message.role {
            MessageRole::System => {
                if !rendered.is_empty() {
                    anyhow::bail!("system message must come before the first turn");
                }
                match &mut system_prefix {
                    Some(existing) => *existing = format!("{existing}\n\n{}", message.content),
                    None => system_prefix = Some(message.content.clone()),
                }
            }
            MessageRole::User => {
                let content = match system_prefix.take() {
                    Some(system) => format!("{system}\n\n{}", message.content),
                    None => message.content.clone(),
                };
                rendered.push(serde_json::json!({ "role": "user", "content": content }));
            }
            MessageRole::Assistant => {
                rendered.push(serde_json::json!({
                    "role": "assistant",
                    "content": message.content,
                }));
            }
            MessageRole::ToolResult(name) => {
                let content = format!("[Tool result for {name}]\n{}", message.content);
                rendered.push(serde_json::json!({ "role": "user", "content": content }));
            }
        }
    }
    anyhow::ensure!(
        !rendered.is_empty(),
        "conversation has no turns after the system message"
    );

    render_value(template, serde_json::Value::Array(rendered))
}

/// Render only the turns that follow a retained assistant reply, for
/// assembling a prompt on top of the KV cache: closes the previous
/// assistant turn, renders the new messages through the template and
/// ends with the generation prompt. The BOS token is omitted (it is
/// already part of the cached prefix).
pub fn render_turn_suffix(template: &str, suffix: &[Message]) -> Result<String> {
    anyhow::ensure!(!suffix.is_empty(), "cannot render an empty turn suffix");
    let rendered = render_messages(template, suffix)?;
    let body = rendered
        .strip_prefix("<bos>")
        .with_context(|| "rendered suffix does not start with the BOS token")?;
    Ok(format!("<end_of_turn>\n{body}"))
}

fn render_value(template: &str, messages: serde_json::Value) -> Result<String> {
    let mut env = Environment::new();
    env.add_template("chat", template)
        .context("failed to parse chat template")?;
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );

    let tmpl = env
        .get_template("chat")
        .context("chat template not found")?;
    tmpl.render(minijinja::context! {
        messages => Value::from_serialize(&messages),
        bos_token => "<bos>",
        add_generation_prompt => true,
    })
    .context("failed to render chat template")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_system_and_prompt() {
        let tmpl =
            "{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}{{ bos_token }}";
        let out = render(tmpl, Some("be brief"), "hi").unwrap();
        assert!(out.contains("[user] be brief\n\nhi"));
        assert!(out.ends_with("<bos>"));
    }

    #[test]
    fn system_none_passes_prompt_through() {
        let tmpl = "{{ messages[0].content }}";
        let out = render(tmpl, None, "plain").unwrap();
        assert!(out.contains("plain"));
        assert!(!out.contains("None"));
    }

    #[test]
    fn bad_template_is_an_error() {
        assert!(render("{% if %}", None, "hi").is_err());
    }

    #[test]
    fn multi_turn_alternates_and_folds_system_into_first_user() {
        let tmpl = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}";
        let out = render_messages(
            tmpl,
            &[
                Message::system("sys"),
                Message::user("q1"),
                Message::assistant("a1"),
                Message::user("q2"),
            ],
        )
        .unwrap();
        assert_eq!(out, "<user>sys\n\nq1<assistant>a1<user>q2");
    }

    #[test]
    fn tool_results_become_framed_user_turns() {
        let tmpl = "{% for m in messages %}<{{ m.role }}>{{ m.content }}\n{% endfor %}";
        let out = render_messages(
            tmpl,
            &[
                Message::user("q"),
                Message::assistant("calling"),
                Message::tool_result("calc", "42"),
            ],
        )
        .unwrap();
        assert!(out.contains("<user>[Tool result for calc]\n42"));
        assert!(!out.contains("ToolResult"));
    }

    #[test]
    fn turn_suffix_closes_previous_assistant_turn_and_starts_generation() {
        let tmpl = "{{ bos_token }}{% for m in messages %}<{{ m.role }}>{{ m.content }}\n{% endfor %}{{ '<start_of_turn>model\n' }}";
        let out = render_turn_suffix(tmpl, &[Message::user("next")]).unwrap();
        assert!(out.starts_with("<end_of_turn>\n"));
        assert!(out.contains("<user>next"));
        assert!(out.ends_with("<start_of_turn>model\n"));
        assert!(!out.starts_with("<bos>"));
    }

    #[test]
    fn system_after_first_turn_is_rejected() {
        let tmpl = "{{ messages | length }}";
        assert!(render_messages(tmpl, &[Message::user("q"), Message::system("late")]).is_err());
    }

    #[test]
    fn system_without_user_turn_is_rejected() {
        let tmpl = "{{ messages | length }}";
        assert!(render_messages(tmpl, &[Message::system("only")]).is_err());
    }
}
