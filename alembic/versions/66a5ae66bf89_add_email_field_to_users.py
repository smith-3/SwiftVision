"""Add email field to users

Revision ID: 66a5ae66bf89
Revises: 6c6b1f29cdbe
Create Date: 2024-11-26 19:54:47.783163

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '66a5ae66bf89'
down_revision = '6c6b1f29cdbe'
branch_labels = None
depends_on = None


def upgrade():
    # Paso 1: Agregar la columna email como nullable
    op.add_column('users', sa.Column('email', sa.String(length=255), nullable=True))

    # Paso 2: Población temporal de la columna email para las filas existentes
    op.execute("UPDATE users SET email = CONCAT('user_', id, '@example.com')")

    # Paso 3: Modificar la columna para que sea NOT NULL y única
    op.alter_column('users', 'email', nullable=False)
    op.create_unique_constraint('uq_users_email', 'users', ['email'])

def downgrade():
    # Revertir los cambios
    op.drop_constraint('uq_users_email', 'users', type_='unique')
    op.drop_column('users', 'email')
