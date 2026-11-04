from pydantic import BaseModel, EmailStr

class LoginSchema(BaseModel):
    email: EmailStr
    motDePasse: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
