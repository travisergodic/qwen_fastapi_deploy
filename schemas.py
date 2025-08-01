from typing import List, Optional, Literal, Union

from pydantic import BaseModel, Field


class TextContent(BaseModel):
    type: Literal["text"]
    text: str

class ImageContent(BaseModel):
    type: Literal["image"]
    image: str  # base64 string

ContentItem = Union[TextContent, ImageContent]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: List[ContentItem]

class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: Optional[int] = Field(default=1000, ge=1, le=4096)
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    do_sample: Optional[bool] = True