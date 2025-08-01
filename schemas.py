from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class ContentItem(BaseModel):
    type: Literal["text", "image"]
    text: Optional[str] = Field(
        default=None,
        description="Text content, required when type == 'text'"
    )
    image: Optional[str] = Field(
        default=None,
        description="Base64-encoded image string, required when type == 'image'"
    )

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: List[ContentItem]

class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: Optional[int] = Field(default=1000, ge=1, le=4096)
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    do_sample: Optional[bool] = True