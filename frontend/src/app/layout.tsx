import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { WorkbenchProvider } from "@/lib/workbench-store";
import { AutoAnnotateProvider } from "@/features/auto-annotate/store";
import { AppShell } from "@/components/AppShell";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "CONTROL ONE — Qwen3-VL Evaluation Workbench",
  description:
    "Local zero-shot & few-shot evaluation workbench for Qwen3-VL MLX LoRA and Ollama models.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} dark`}
    >
      <body className="min-h-screen bg-[#0A0A0C] text-white selection:bg-[#F2853F]/30 selection:text-white">
        <WorkbenchProvider>
          <AutoAnnotateProvider>
            <AppShell>{children}</AppShell>
          </AutoAnnotateProvider>
        </WorkbenchProvider>
      </body>
    </html>
  );
}
