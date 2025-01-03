// components/ThemeToggleButton.js
"use client";

import { useState, useEffect } from "react";
import { Button } from '@/components/ui/button';
import { Sun, Moon } from "lucide-react"; // Icons for light/dark modes

export function ModeToggle () {
  const [theme, setTheme] = useState("light");

  useEffect(() => {
    // Set initial theme from localStorage or default to light
    const storedTheme = localStorage.getItem("theme") || "light";
    setTheme(storedTheme);
    document.documentElement.classList.toggle("dark", storedTheme === "dark");
  }, []);

  const toggleTheme = () => {
    const newTheme = theme === "light" ? "dark" : "light";
    setTheme(newTheme);
    document.documentElement.classList.toggle("dark", newTheme === "dark");
    localStorage.setItem("theme", newTheme); // Persist theme
  };

  return (
    <Button
      onClick={toggleTheme}
      variant="ghost" // Styling variant from shadcn/ui
      size="sm" // Small button
      className="flex items-center gap-2"
    >
      {theme === "light" ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
    </Button>
  );
};

