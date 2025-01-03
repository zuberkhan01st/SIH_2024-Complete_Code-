import React from "react";
import { Button } from "@react-email/button";
import { Html } from "@react-email/html";
import { Heading } from "@react-email/heading";
import { Text } from "@react-email/text";

export default function MagicLinkEmail({ params }) {
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
            Here is your magic link for login. This link will be valid only for 15 minutes.
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
            Click Here
          </Button>
          <Text
            style={{
              fontSize: "14px",
              color: "#777777",
              marginTop: "20px",
              textAlign: "center",
            }}
          >
            If you didn't request this email, please ignore it.
          </Text>
        </div>
      </body>
    </Html>
  );
}
