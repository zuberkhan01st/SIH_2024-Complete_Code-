import React, { useEffect, useRef } from 'react';

const CursorTrail = ({ color = "#90EE90" }) => { 
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    // Adjust the canvas size to fill the viewport
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const cursorTrail = (props) => {
      const { ref, color } = props;
      const ctx = ref.current?.getContext('2d');
      let AnimationFeature = {
        friction: 0.98, // High friction for fast movement
        trails: 10, // Few particles
        size: 2, // Ultra-small particle size
        dampening: 0.98, // High dampening
        tension: 0.99, // High tension
        speed: 5, // Very high speed
        lifespan: 200 // Particles last for 200ms
      };

      let cursorPosition = { x: 0, y: 0 };
      let running = true;
      let particles = [];

      class Particle {
        x = 0;
        y = 0;
        vx = 0;
        vy = 0;
        creationTime = 0;

        constructor(x, y) {
          this.x = x;
          this.y = y;
          this.vx = (Math.random() - 0.5) * AnimationFeature.speed;
          this.vy = (Math.random() - 0.5) * AnimationFeature.speed;
          this.creationTime = Date.now();
        }

        update() {
          this.x += this.vx;
          this.y += this.vy;
          this.vx *= AnimationFeature.dampening;
          this.vy *= AnimationFeature.dampening;
        }

        draw() {
          ctx.beginPath();
          ctx.arc(this.x, this.y, AnimationFeature.size, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.closePath();
        }

        isExpired() {
          return Date.now() - this.creationTime > AnimationFeature.lifespan;
        }
      }

      function renderAnimation() {
        if (running) {
          ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
          ctx.globalCompositeOperation = "lighter";
          particles.forEach(particle => {
            particle.update();
            particle.draw();
          });
          // Remove expired particles
          particles = particles.filter(p => !p.isExpired());
          window.requestAnimationFrame(renderAnimation);
        }
      }

      function move(event) {
        if (event instanceof MouseEvent) {
          cursorPosition.x = event.clientX;
          cursorPosition.y = event.clientY;
        } else if (event instanceof TouchEvent) {
          cursorPosition.x = event.touches[0].pageX;
          cursorPosition.y = event.touches[0].pageY;
        }
        event.preventDefault();
        // Create new particles at cursor position
        for (let i = 0; i < AnimationFeature.trails; i++) {
          particles.push(new Particle(cursorPosition.x, cursorPosition.y));
        }
      }

      function createParticle(event) {
        if (event.touches.length === 1) {
          cursorPosition.x = event.touches[0].pageX;
          cursorPosition.y = event.touches[0].pageY;
        }
      }

      function onMouseMove(e) {
        document.removeEventListener("mousemove", onMouseMove);
        document.removeEventListener("touchstart", onMouseMove);
        document.addEventListener("mousemove", move);
        document.addEventListener("touchmove", createParticle);
        document.addEventListener("touchstart", createParticle);
        move(e);
        renderAnimation();
      }

      function resizeCanvas() {
        ctx.canvas.width = window.innerWidth;
        ctx.canvas.height = window.innerHeight;
      }

      function stopAnimation() {
        running = false;
      }

      function startAnimation() {
        if (!running) {
          running = true;
          renderAnimation();
        }
      }

      function renderTrailCursor() {
        document.addEventListener("mousemove", onMouseMove);
        document.addEventListener("touchstart", onMouseMove);
        window.addEventListener("orientationchange", resizeCanvas);
        window.addEventListener("resize", resizeCanvas);
        window.addEventListener("focus", startAnimation);
        window.addEventListener("blur", stopAnimation);
        resizeCanvas();
      }

      function cleanUp() {
        document.removeEventListener("mousemove", move);
        document.removeEventListener("touchmove", createParticle);
        document.removeEventListener("touchstart", createParticle);
        document.removeEventListener("mousemove", onMouseMove);
        document.removeEventListener("touchstart", onMouseMove);
        window.removeEventListener("orientationchange", resizeCanvas);
        window.removeEventListener("resize", resizeCanvas);
        window.removeEventListener("focus", startAnimation);
        window.removeEventListener("blur", stopAnimation);
      }

      return { cleanUp, renderTrailCursor, stopAnimation, startAnimation };
    };

    const { cleanUp, renderTrailCursor } = cursorTrail({
      ref: canvasRef,
      color,
    });

    renderTrailCursor();

    return () => {
      cleanUp();
    };
  }, [color]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed top-0 left-0 w-full h-full pointer-events-none"
    ></canvas>
  );
};

export default CursorTrail;
