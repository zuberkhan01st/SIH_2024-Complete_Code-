"use client";
import React from "react";
import { signOut } from "next-auth/react";
import { LogOut } from "lucide-react";

export default function SignoutButton({ type }) {
  return (
    <div>
      <button
        className="text-red-500"
        onClick={() =>
          signOut({
            callbackUrl: type === "Admin" ? "/admin/login" : "/",
            redirect: true,
          })
        }
      >
      <div className="flex items-center px-4 py-2 hover:bg-gray-700 dark:hover:bg-gray-600 rounded-md"><LogOut className="w-5 h-5 mr-3"/> Logout</div>
       
      </button>
    </div>
  );
}
