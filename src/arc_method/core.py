import asyncio
from groq import AsyncGroq
from openai import OpenAI


__all__ = ["ArcMethod", "help"]


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
        
    

    @staticmethod
    def print_help():
        """
        Displays usage instructions for the ArcMethod plugin.
        """
        help_text = """
        ===========================
        ArcMethod - Usage Guide
        ===========================

        ArcMethod is an LLM interaction framework based on the Arc Method —
        a philosophy where you describe the problem to the AI instead of writing
        traditional algorithms.

        The class supports two providers:
        • OpenAI  (GPT-4.1, GPT-4.1-mini, etc.)
        • Groq    (LLaMA 3.3 70B Versatile)

        --------------------------------------
        1. Installing the Package
        --------------------------------------

        Install from pip:
            pip install arc-method

        --------------------------------------
        2. Basic Initialization
        --------------------------------------

        Using OpenAI:
            from arc_method.core import ArcMethod
            
            arc = ArcMethod(
                provider="openai",
                api_key="YOUR_OPENAI_KEY"
            )

        Using Groq LLaMA:
            arc = ArcMethod(
                provider="llama",
                api_key="YOUR_GROQ_KEY"
            )

        --------------------------------------
        3. Main Interaction Function
        --------------------------------------

        ArcMethod uses a single unified function:

            interact(system_prompt: str, user_prompt: str) -> str

        Example:

            system = "You are an AI using the Arc Method."
            user   = "Extract the GUID from this text: TRT-REF-9988-ABCD"

            response = arc.interact(system, user)
            print(response)

        --------------------------------------
        4. Philosophy of the Arc Method
        --------------------------------------

        Arc Method is based on 3 principles:

        • You don't write complex algorithms.
            You describe the problem to the AI clearly.

        • The system is flexible.
            Format changes will not break your logic.

        • AI performs interpretation close to human reasoning.
            This reduces development time and errors.

        --------------------------------------
        5. Model Selection
        --------------------------------------

        OpenAI provider uses:
            gpt-4.1-mini (default)
        You may override with any OpenAI chat model.

        Groq provider uses:
            llama-3.3-70b-versatile (default)
        This is extremely fast and cost-efficient.

        --------------------------------------
        6. Error Notes
        --------------------------------------

        If invalid provider:
            ValueError: provider must be 'openai' or 'llama'

        If API key missing/invalid:
            The underlying SDKs will raise authentication errors.

        --------------------------------------
        7. About the License
        --------------------------------------

        ArcMethod is licensed under the ARC License v1.1.
        Commercial and military use is restricted unless permission
        is granted by the copyright holder.

        --------------------------------------
        End of Help
        --------------------------------------
        """
        print(help_text)


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


def help():
    ArcMethod.print_help()