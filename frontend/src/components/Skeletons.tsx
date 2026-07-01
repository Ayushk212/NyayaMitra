"use client";

export function SearchSkeleton() {
  return (
    <div className="space-y-4">
      {[...Array(5)].map((_, i) => (
        <div key={i} className="card p-5 space-y-3">
          <div className="skeleton h-5 w-3/4" />
          <div className="flex gap-3">
            <div className="skeleton h-4 w-24" />
            <div className="skeleton h-4 w-20" />
            <div className="skeleton h-4 w-16" />
          </div>
          <div className="skeleton h-4 w-full" />
          <div className="skeleton h-4 w-5/6" />
        </div>
      ))}
    </div>
  );
}

export function ChatSkeleton() {
  return (
    <div className="space-y-2">
      <div className="flex gap-2 items-center text-sm text-[var(--text-secondary)]">
        <div className="w-5 h-5 rounded-full bg-orange-100 animate-pulse" />
        <span>Analyzing...</span>
      </div>
      <div className="card p-4 space-y-2">
        <div className="skeleton h-4 w-full" />
        <div className="skeleton h-4 w-4/5" />
        <div className="skeleton h-4 w-3/4" />
      </div>
    </div>
  );
}

export function CaseSkeleton() {
  return (
    <div className="space-y-4 pt-4">
      <div className="skeleton h-8 w-2/3" />
      <div className="flex gap-3">
        <div className="skeleton h-5 w-32" />
        <div className="skeleton h-5 w-24" />
      </div>
      <div className="space-y-3 mt-6">
        {[...Array(8)].map((_, i) => (
          <div key={i} className="skeleton h-4 w-full" />
        ))}
      </div>
    </div>
  );
}
