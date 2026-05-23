pip install asyncpg==0.29.0 sqlalchemy[asyncio]==2.0.23 --break-system-packages

kubectl port-forward -n default svc/postgres-pooler 5432:5432 &
sleep 2
python3 src/tests/simulate_company/create_fake_tables.py
