import React, { Suspense } from "react";
import { AutoAnnotateView } from "@/features/auto-annotate/AutoAnnotateView";

export default function AutoAnnotatePage() {
  return (
    <Suspense
      fallback={
        <div className="c1-card p-12 text-center text-sm text-[#A1A1AA]">
          Loading Auto-Annotate workspace...
        </div>
      }
    >
      <AutoAnnotateView />
    </Suspense>
  );
}
