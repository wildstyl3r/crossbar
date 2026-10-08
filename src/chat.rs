use anyhow::{Context, Result};
use minijinja::{Environment, Value};

pub fn render(template: &str, system: Option<&str>, prompt: &str) -> Result<String> {
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

    let content = match system {
        Some(system) => format!("{system}\n\n{prompt}"),
        None => prompt.to_string(),
    };
    let messages = serde_json::json!([{ "role": "user", "content": content }]);

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
}
