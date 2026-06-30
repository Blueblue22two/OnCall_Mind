from langchain_core.messages import AIMessage

from tests.evaluation.evaluate_rag import _generate_answers


class _FakeModel:
    def __init__(self) -> None:
        self.messages = []

    async def ainvoke(self, messages):
        self.messages.append(messages)
        return AIMessage(content="基于上下文生成的答案")


async def test_generate_answers_uses_fixed_retrieval_contexts(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(
        "app.core.llm_factory.create_chat_qwen",
        lambda **_: model,
    )

    answers = await _generate_answers(["如何排查？"], [["先检查 CPU。", "再检查进程。"]])

    assert answers == ["基于上下文生成的答案"]
    prompt = model.messages[0][1].content
    assert "如何排查？" in prompt
    assert "先检查 CPU。" in prompt
    assert "再检查进程。" in prompt
