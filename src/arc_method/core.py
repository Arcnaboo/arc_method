import asyncio
from groq import AsyncGroq
from openai import OpenAI


class ArcMethod:
    """
    ArcMethod - LLM tabanlı metin işleme ve yorumlama sınıfı.
    provider: "openai" veya "llama"
    """

    def __init__(self, provider: str, api_key: str, model: str = None):
        self.provider = provider.lower()
        self.api_key = api_key

        if self.provider == "openai":
            self.model = model or "gpt-4.1-mini"
            self.client = OpenAI(api_key=self.api_key)

        elif self.provider == "llama":
            self.model = model or "llama-3.3-70b-versatile"
            # Groq client
            self.client = AsyncGroq(api_key=self.api_key)

        else:
            raise ValueError("provider must be 'openai' or 'llama'")

    # -----------------------
    # Internal async wrappers
    # -----------------------

    async def _invoke_openai(self, system_prompt: str, user_prompt: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    async def _invoke_groq(self, system_prompt: str, user_prompt: str):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    # -----------------------
    # Public ArcMethod API
    # -----------------------

    def interact(self, system_prompt: str, user_prompt: str) -> str:
        """
        Arc Yöntemi etkileşim fonksiyonu.
        Tek bir system prompt + user prompt alır.
        Provider'a göre doğru LLM'i kullanır.
        """

        if self.provider == "openai":
            return asyncio.run(self._invoke_openai(system_prompt, user_prompt))

        elif self.provider == "llama":
            return asyncio.run(self._invoke_groq(system_prompt, user_prompt))

        else:
            raise RuntimeError("Invalid provider in interact()")
