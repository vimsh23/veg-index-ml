# Frontend Dockerfile — serves static HTML via nginx
FROM nginx:alpine

# Copy the app
COPY index.html /usr/share/nginx/html/index.html

# Nginx config: proxy /api → backend
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
