import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3001";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const origin = `${protocol}://${host}`;
  const title = "Sona — Find your note";
  const description =
    "A focused singing studio with guided vocal training, responsive piano exercises, live pitch feedback, and song practice.";

  return {
    metadataBase: new URL(origin),
    title,
    description,
    openGraph: {
      type: "website",
      url: origin,
      title,
      description,
      images: [{ url: `${origin}/og.png`, width: 1536, height: 1024, alt: "Sona pitch lane meeting a glowing in-tune note" }],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description,
      images: [`${origin}/og.png`],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
