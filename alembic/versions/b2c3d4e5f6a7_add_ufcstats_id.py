"""add_ufcstats_id_to_fighters_and_fights

Revision ID: b2c3d4e5f6a7
Revises: 3214403b9d00
Create Date: 2026-03-20 01:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = '3214403b9d00'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('fighters', sa.Column('ufcstats_id', sa.String(length=64), nullable=True))
    op.create_index('ix_fighters_ufcstats_id', 'fighters', ['ufcstats_id'], unique=True)

    op.add_column('fights', sa.Column('ufcstats_id', sa.String(length=64), nullable=True))
    op.create_index('ix_fights_ufcstats_id', 'fights', ['ufcstats_id'], unique=True)


def downgrade() -> None:
    op.drop_index('ix_fights_ufcstats_id', table_name='fights')
    op.drop_column('fights', 'ufcstats_id')

    op.drop_index('ix_fighters_ufcstats_id', table_name='fighters')
    op.drop_column('fighters', 'ufcstats_id')
