import React from "react";
import { Button } from "@react-email/button";
import { Html } from "@react-email/html";
import { Heading } from "@react-email/heading";
import { Text } from "@react-email/text";

export default function ForgotPasswordEmail({ params }) {
  return (
    <Html>
      <body
        style={{
          fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
          margin: 0,
          padding: 0,
          backgroundColor: "#f4f4f4",
        }}
      >
        <div
          style={{
            maxWidth: "600px",
            margin: "0 auto",
            padding: "20px",
            backgroundColor: "#ffffff",
            borderRadius: "8px",
            boxShadow: "0 0 10px rgba(0, 0, 0, 0.1)",
          }}
        >
          <Heading
            as="h2"
            style={{
              fontSize: "24px",
              fontWeight: "600",
              color: "#333333",
              margin: "0 0 16px",
            }}
          >
            Hello {params.name},
          </Heading>
          <Text
            style={{
              fontSize: "16px",
              color: "#555555",
              lineHeight: "1.5",
              marginBottom: "20px",
            }}
          >
            We received a request to reset your password. If you did not make this request, please ignore this email.
          </Text>
          <Button
            pX={20}
            pY={12}
            href={params.url}
            style={{
              display: "inline-block",
              backgroundColor: "#007bff",
              color: "#ffffff",
              textDecoration: "none",
              borderRadius: "4px",
              fontWeight: "600",
              fontSize: "16px",
              textAlign: "center",
              border: "none",
              boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
            }}
          >
            Reset Password
          </Button>
          <Text
            style={{
              fontSize: "14px",
              color: "#777777",
              marginTop: "20px",
              textAlign: "center",
            }}
          >
            If you did not request a password reset, please ignore this email.
          </Text>
        </div>
      </body>
    </Html>
  );
}
