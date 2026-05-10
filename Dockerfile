FROM manga_translator:v2

COPY ./manga_translator /app/manga_translator


# Add /app to Python module path
ENV PYTHONPATH="/app"

WORKDIR /app

ENTRYPOINT ["python3.12"]