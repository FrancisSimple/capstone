from typing import Optional
from uuid import UUID
from sqlalchemy.dialects.postgresql import UUID as pgUUID
from typing import TYPE_CHECKING
from sqlalchemy import ForeignKey
from src.models.base import TimestampedModel, UUIDModel
from sqlalchemy.orm import Mapped,mapped_column, relationship

#model imports
# if TYPE_CHECKING:
#     from src.objects.user import Student
#     from src.objects.agents import Provider
#     from src.objects.opportunities import Opportunity
#     from src.objects.universities import University
#     from src.objects.scholarships import Scholarship


class Image(UUIDModel, TimestampedModel):
    url: Mapped[str] = mapped_column(nullable = False)
    public_id: Mapped[str] = mapped_column(nullable = False)  # Helps in deleting later
    # foreign keys
    # student_id: Mapped[Optional[UUID]] = mapped_column(pgUUID(as_uuid=True), ForeignKey("students.id"), nullable=True)
    # provider_id: Mapped[Optional[UUID]] = mapped_column(pgUUID(as_uuid=True), ForeignKey("providers.id"), nullable=True)
    # opportunity_id: Mapped[Optional[UUID]] = mapped_column(pgUUID(as_uuid=True), ForeignKey("opportunities.id"), nullable=True)
    # university_id: Mapped[Optional[UUID]] = mapped_column(pgUUID(as_uuid=True), ForeignKey("universities.id"), nullable=True)
    # scholarship_id: Mapped[Optional[UUID]] = mapped_column(pgUUID(as_uuid=True), ForeignKey("scholarships.id"), nullable=True)

    # #relationships
    # student: Mapped[Optional["Student"]] = relationship(back_populates="images", lazy="selectin")
    # provider: Mapped[Optional["Provider"]] = relationship(back_populates="images", lazy="selectin")
    # opportunity:Mapped[Optional["Opportunity"]] = relationship(back_populates="images", lazy="selectin")
    # university: Mapped[Optional["University"]] = relationship(back_populates="images", lazy="selectin")
    # scholarship: Mapped[Optional["Scholarship"]] = relationship(back_populates="images", lazy="selectin")