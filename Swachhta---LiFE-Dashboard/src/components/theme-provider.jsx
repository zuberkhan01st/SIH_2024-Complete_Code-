"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";

// This component wraps the NextThemesProvider to provide theme management
export function ThemeProvider({ children, ...props }) {
  return <NextThemesProvider {...props}>{children}</NextThemesProvider>;
}
