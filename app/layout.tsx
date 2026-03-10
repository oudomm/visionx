import type { Metadata, Viewport } from "next";
import { Geist_Mono, Noto_Sans_Khmer } from "next/font/google";
import "./globals.css";

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const notoSansKhmer = Noto_Sans_Khmer({
  variable: "--font-khmer",
  subsets: ["khmer"],
  weight: ["400", "700"],
});

export const metadata: Metadata = {
  title: "VisionKH — AI Object Detection in Khmer",
  description: "Real-time AI object detection app that identifies objects through your camera and speaks their names in Khmer (ភាសាខ្មែរ). Powered by TensorFlow.js and COCO-SSD. Free, no API key, runs entirely in your browser.",
  keywords: ["AI object detection", "Khmer", "ភាសាខ្មែរ", "TensorFlow.js", "real-time", "camera", "COCO-SSD"],
  authors: [{ name: "Oudom", url: "https://oudom.dev" }],
  openGraph: {
    title: "VisionKH — AI Object Detection in Khmer",
    description: "Point your camera at anything. VisionKH identifies it and speaks its name in Khmer.",
    url: "https://visionkh.oudom.dev",
    siteName: "VisionKH",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "VisionKH — AI Object Detection in Khmer",
    description: "Point your camera at anything. VisionKH identifies it and speaks its name in Khmer.",
    creator: "@oudom",
  },
  metadataBase: new URL("https://visionkh.oudom.dev"),
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="km">
      <head>
        <link rel="preconnect" href="https://storage.googleapis.com" />
        <link rel="dns-prefetch" href="https://storage.googleapis.com" />
      </head>
      <body
        className={`${geistMono.variable} ${notoSansKhmer.variable} antialiased`}
        suppressHydrationWarning
      >
        {children}
      </body>
    </html>
  );
}
