from app.modules.db.test_database import Base, engine
import app.modules.db.models 

def init_db():
    Base.metadata.create_all(bind=engine)
    print("[INFO] Database initialized (Postgres).")

if __name__ == "__main__":
    init_db()
