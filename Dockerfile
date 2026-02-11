# Dockerfile for Railway (fallback if Nixpacks + LibreOffice is problematic)
FROM node:20-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice libreoffice-writer libreoffice-calc libreoffice-impress \
    poppler-utils ghostscript fontconfig fonts-dejavu-core fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev
COPY server.js ./

EXPOSE 3000
ENV PORT=3000
CMD ["node", "server.js"]
