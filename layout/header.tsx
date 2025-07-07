"use client";

import { Sparkles, Star } from "lucide-react";

export default function Header() {
  return (
    <div className="flex items-center justify-between p-2 px-4">
      <div className="flex items-center gap-2">
        <Sparkles className="size-4" />
        <h1 className="font-semibold">Camara Chat Bot</h1>
      </div>
      <div className="flex items-center justify-end gap-4">
        <div className="flex items-center gap-2">
          <a
            href="https://camaraproject.org/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200"
          >
            CamaraBot 
          </a>
          <img
            className="h-[24px] w-[24px] rounded-sm"
            src="/camara.jpeg"
            alt="Llama Logo"
          />
        </div>
       
      </div>
    </div>
  );
}
