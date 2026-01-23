from typing import List
from sqlalchemy import text


class SQLHelper:
    """
    A utility class that builds raw SQL query strings dynamically.
    Each static method returns a SQL statement that can be executed
    by SQLAlchemy.
    """

    @staticmethod
    def raw(query: str):
        """
        Wrap a raw SQL string inside SQLAlchemy's `text()` function.
        - This makes it executable in the session.
        """
        return text(query)

    @staticmethod
    def select_all(table: str):
        """
        Build a SQL query that selects all rows from a table.
        Example: SELECT * FROM users
        """
        return text(f"SELECT * FROM {table}")

    @staticmethod
    def insert(table: str, columns: List[str]):
        """
        Build an INSERT query with placeholders for values.
        Example:
            columns = ["name", "age"]
            -> INSERT INTO users (name, age) VALUES (:name, :age)

        - `:{col}` are placeholders for parameters that we pass later.
        """
        cols = ", ".join(columns)
        values = ", ".join([f":{col}" for col in columns])
        return text(f"INSERT INTO {table} ({cols}) VALUES ({values})")

    @staticmethod
    def update(table: str, columns: List[str], lookup_field: str):
        """
        Build an UPDATE query.
        Example:
            table = "users"
            columns = ["name", "age"]
            lookup_field = "id"
            -> UPDATE users SET name = :name, age = :age WHERE id = :id
        """
        set_clause = ", ".join([f"{col} = :{col}" for col in columns])
        return text(f"UPDATE {table} SET {set_clause} WHERE {lookup_field} = :{lookup_field}")

    @staticmethod
    def delete(table: str, field_name: str):
        """
        Build a DELETE query.
        Example:
            DELETE FROM users WHERE id = :id
        """
        return text(f"DELETE FROM {table} WHERE {field_name} = :{field_name}")
