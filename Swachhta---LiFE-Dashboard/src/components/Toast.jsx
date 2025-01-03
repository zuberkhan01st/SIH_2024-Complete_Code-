"use client";
import React, { useEffect } from "react";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { useSearchParams } from "next/navigation";

export default function Toast() {
  const params = useSearchParams();

  useEffect(() => {
    const error = params?.get("error");
    if (error && error !== "") {
      toast.error(error, { theme: "colored" });
    }
  }, [params]);

  return (
    <div>
      <ToastContainer />
    </div>
  );
}
