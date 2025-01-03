export async function GET(request) {
    try {
      const response = await fetch(
        "https://life-practices-backend.onrender.com/api/get_data?lat=21.1458&lon=79.0882"
      );
  
      if (!response.ok) {
        return new Response(
          JSON.stringify({ error: "Failed to fetch weather data." }),
          { status: response.status }
        );
      }
  
      const data = await response.json();
      return new Response(JSON.stringify(data), { status: 200 });
    } catch (error) {
      console.error("API Error:", error);
      return new Response(JSON.stringify({ error: "Internal server error." }), {
        status: 500,
      });
    }
  }
  