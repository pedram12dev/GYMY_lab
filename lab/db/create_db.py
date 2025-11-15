# lab/db/create_db.py
from lab.db.test_database import Base, engine


def init_db():
    Base.metadata.create_all(bind=engine)
    print("[INFO] Test database initialized.")


if __name__ == "__main__":
    init_db()
