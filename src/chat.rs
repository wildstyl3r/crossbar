use anyhow::{Context, Result};
use minijinja::{Environment, Value};

pub fn render(template: &str, system: Option<&str>, prompt: &str) -> Result<String> {
    let mut env = Environment::new();
    env.add_template("chat", template)
        .context("failed to parse chat template")?;
    env.add_function("raise_exception", |msg: String| -> Result<Value, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
    });

    let content = match system {
        Some(system) => format!("{system}\n\n{prompt}"),
        None => prompt.to_string(),
    };
    let messages = serde_json::json!([{ "role": "user", "content": content }]);

    let tmpl = env.get_template("chat").context("chat template not found")?;
    tmpl.render(minijinja::context! {
        messages => Value::from_serialize(&messages),
        bos_token => "<bos>",
        add_generation_prompt => true,
    })
    .context("failed to render chat template")
}
