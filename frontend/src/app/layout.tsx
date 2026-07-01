import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NyayaMitra — AI-Powered Indian Legal Research",
  description:
    "Search Indian case law, get AI-powered legal analysis with citations. Supreme Court & High Court judgments at your fingertips.",
  keywords: ["Indian law", "case law", "legal AI", "Supreme Court", "legal research"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Caveat:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}
