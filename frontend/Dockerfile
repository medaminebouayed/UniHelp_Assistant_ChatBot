# ── Build stage ──
FROM node:18-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --legacy-peer-deps

COPY . .

# Override environment for Docker (API calls via localhost:8000)
RUN npm run build -- --configuration production

# ── Serve stage ──
FROM nginx:1.25-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=build /app/dist/uni-help /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
