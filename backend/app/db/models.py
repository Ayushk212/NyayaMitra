"""SQLAlchemy models for cases and case chunks."""
from sqlalchemy import Column, Integer, String, Text, Date, ForeignKey, Float
from sqlalchemy.orm import relationship
from app.db.database import Base


class Case(Base):
    __tablename__ = "cases"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False, index=True)
    court = Column(String, nullable=False, index=True)
    date = Column(String, nullable=True)
    judges = Column(Text, nullable=True)  # JSON array as string
    case_type = Column(String, nullable=True, index=True)
    citation = Column(String, nullable=True)
    full_text = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)

    chunks = relationship("CaseChunk", back_populates="case", cascade="all, delete-orphan")


class CaseChunk(Base):
    __tablename__ = "case_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String, ForeignKey("cases.id"), nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    paragraph_ref = Column(String, nullable=True)
    embedding_index = Column(Integer, nullable=True)  # index position in FAISS

    case = relationship("Case", back_populates="chunks")
