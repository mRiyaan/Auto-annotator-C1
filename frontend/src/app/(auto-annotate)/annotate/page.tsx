import React, { Suspense } from "react";
import { ManualAnnotatorView } from "@/features/auto-annotate/ManualAnnotatorView";

export default function ManualAnnotatorPage() {
  return (
    <Suspense
      fallback={
        <div className="p-8 font-mono-tech text-xs text-[#8A8A8E]">
          Loading Manual Annotator Studio...
        </div>
      }
    >
      <ManualAnnotatorView />
    </Suspense>
  );
}
