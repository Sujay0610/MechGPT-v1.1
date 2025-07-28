/** @type {import('next').NextConfig} */
const nextConfig = {
  // App directory is enabled by default in Next.js 14
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/api/:path*',
      },
    ]
  },
  // Add experimental features for better proxy handling
  experimental: {
    proxyTimeout: 30000,
  },
}

module.exports = nextConfig