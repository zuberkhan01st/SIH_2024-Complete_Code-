import { Menu } from "lucide-react";
import React from "react";
import {ModeToggle} from '@/components/ui/modeToggle'
import Notification from "./Notification"

export default function Header({ toggleSidebar }) {
  return (
    <header className="bg-red-600 text-white p-4 md:ml-64 shadow-md dark:bg-red-800 flex items-center justify-between">
      <button
        onClick={toggleSidebar}
        className="text-white md:hidden"
      >
        <Menu/>
      </button>
      <h1 className="text-xl">Swachhta & LiFE Dashboard</h1>
      <div className="flex items-center justify-center gap-6">
      <Notification/>
      <ModeToggle/>
      </div>
    </header>
  );
}
