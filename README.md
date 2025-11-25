# ArcMethod

ArcMethod is an LLM interaction framework based on the Arc Yöntemi.

## Installation

```
pip install arc-method
```

## Usage

```python
from arc_method.core import ArcMethod

arc = ArcMethod(provider="llama", api_key="YOUR_GROQ_KEY")

system = "You are an AI assistant using the Arc Method."
user = "Extract the GUID from this text: TRT-REF-9988-ABCD"

print(arc.interact(system, user))
```

## Providers
- **openai** → GPT-4.1 family  
- **llama** → Groq LLaMA 3.3 70B Versatile

## License
MIT
