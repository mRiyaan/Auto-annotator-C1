import React, { Suspense } from "react";
import { VerifyQueueView } from "@/features/auto-annotate/VerifyQueueView";

export default function VerifyAutoAnnotationPage() {
  return (
    <Suspense
      fallback={
        <div className="c1-card p-12 text-center text-sm text-[#A1A1AA]">
          Loading Verify Auto-Annotation workspace...
        </div>
      }
    >
      <VerifyQueueView />
    </Suspense>
  );
}
