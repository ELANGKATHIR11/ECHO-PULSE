import React, { useRef, useEffect } from 'react';
import { useTheme } from '../../context/ThemeContext';

interface OceanLiquidCausticBackgroundProps {
  interactive?: boolean;
}

export const OceanLiquidCausticBackground: React.FC<OceanLiquidCausticBackgroundProps> = ({
  interactive = true,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { isDark } = useTheme();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) return;

    let animationFrameId: number;
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };

    window.addEventListener('resize', handleResize);

    // Mouse interactivity
    const mouse = {
      x: width * 0.5,
      y: height * 0.3,
      targetX: width * 0.5,
      targetY: height * 0.3,
      radius: 180,
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!interactive) return;
      mouse.targetX = e.clientX;
      mouse.targetY = e.clientY;
    };

    if (interactive) {
      window.addEventListener('mousemove', handleMouseMove, { passive: true });
    }

    // 3D-like floating marine acoustic particles / plankton / bubbles
    const particleCount = 45;
    const particles = Array.from({ length: particleCount }, () => ({
      x: Math.random() * width,
      y: Math.random() * height,
      z: Math.random() * 2 + 0.5, // depth factor for 3D parallax
      radius: Math.random() * 2.2 + 0.8,
      speedY: -(Math.random() * 0.35 + 0.15),
      speedX: (Math.random() - 0.5) * 0.3,
      alpha: Math.random() * 0.5 + 0.2,
      pulsePhase: Math.random() * Math.PI * 2,
    }));

    // Floating refraction caustic waves parameters
    let time = 0;

    const render = () => {
      time += 0.016;

      // Smooth mouse follow
      mouse.x += (mouse.targetX - mouse.x) * 0.05;
      mouse.y += (mouse.targetY - mouse.y) * 0.05;

      ctx.clearRect(0, 0, width, height);

      // 1. Fluid Caustic Waves (Layer 1 - deep ocean glow)
      const grad1 = ctx.createRadialGradient(
        width * 0.25 + Math.sin(time * 0.4) * 80,
        height * 0.2 + Math.cos(time * 0.3) * 60,
        20,
        width * 0.25,
        height * 0.2,
        width * 0.55
      );

      if (isDark) {
        grad1.addColorStop(0, 'rgba(6, 182, 212, 0.16)');
        grad1.addColorStop(0.5, 'rgba(3, 105, 161, 0.07)');
        grad1.addColorStop(1, 'rgba(2, 6, 23, 0)');
      } else {
        grad1.addColorStop(0, 'rgba(56, 189, 248, 0.28)');
        grad1.addColorStop(0.5, 'rgba(14, 165, 233, 0.12)');
        grad1.addColorStop(1, 'rgba(210, 236, 249, 0)');
      }

      ctx.fillStyle = grad1;
      ctx.fillRect(0, 0, width, height);

      // 2. Fluid Caustic Waves (Layer 2 - secondary oceanic drift)
      const grad2 = ctx.createRadialGradient(
        width * 0.75 + Math.cos(time * 0.35) * 90,
        height * 0.75 + Math.sin(time * 0.45) * 70,
        40,
        width * 0.75,
        height * 0.75,
        width * 0.5
      );

      if (isDark) {
        grad2.addColorStop(0, 'rgba(14, 116, 144, 0.14)');
        grad2.addColorStop(0.6, 'rgba(8, 47, 73, 0.06)');
        grad2.addColorStop(1, 'rgba(2, 6, 23, 0)');
      } else {
        grad2.addColorStop(0, 'rgba(125, 211, 252, 0.30)');
        grad2.addColorStop(0.6, 'rgba(186, 230, 253, 0.15)');
        grad2.addColorStop(1, 'rgba(210, 236, 249, 0)');
      }

      ctx.fillStyle = grad2;
      ctx.fillRect(0, 0, width, height);

      // 3. Interactive Liquid Caustic Specular Refraction around Cursor
      if (interactive) {
        const mouseGrad = ctx.createRadialGradient(
          mouse.x,
          mouse.y,
          0,
          mouse.x,
          mouse.y,
          mouse.radius * 1.5
        );
        if (isDark) {
          mouseGrad.addColorStop(0, 'rgba(34, 211, 238, 0.15)');
          mouseGrad.addColorStop(0.4, 'rgba(6, 182, 212, 0.06)');
          mouseGrad.addColorStop(1, 'rgba(2, 7, 18, 0)');
        } else {
          mouseGrad.addColorStop(0, 'rgba(2, 132, 199, 0.20)');
          mouseGrad.addColorStop(0.5, 'rgba(56, 189, 248, 0.08)');
          mouseGrad.addColorStop(1, 'rgba(255, 255, 255, 0)');
        }
        ctx.fillStyle = mouseGrad;
        ctx.fillRect(0, 0, width, height);
      }

      // 4. Draw 3D Depth Floating Hydro-Acoustic Luminous Specular Particles
      particles.forEach((p) => {
        p.y += p.speedY * p.z;
        p.x += p.speedX * p.z + Math.sin(time + p.pulsePhase) * 0.2;

        // Wrap-around
        if (p.y < -20) {
          p.y = height + 20;
          p.x = Math.random() * width;
        }
        if (p.x < -20) p.x = width + 20;
        if (p.x > width + 20) p.x = -20;

        const currentAlpha =
          p.alpha * (0.6 + 0.4 * Math.sin(time * 2 + p.pulsePhase));

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius * p.z, 0, Math.PI * 2);

        if (isDark) {
          ctx.fillStyle = `rgba(34, 211, 238, ${currentAlpha * 0.75})`;
          ctx.shadowColor = 'rgba(34, 211, 238, 0.6)';
          ctx.shadowBlur = 8 * p.z;
        } else {
          ctx.fillStyle = `rgba(2, 132, 199, ${currentAlpha * 0.65})`;
          ctx.shadowColor = 'rgba(2, 132, 199, 0.4)';
          ctx.shadowBlur = 6 * p.z;
        }

        ctx.fill();
        ctx.shadowBlur = 0; // reset
      });

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener('resize', handleResize);
      if (interactive) {
        window.removeEventListener('mousemove', handleMouseMove);
      }
      cancelAnimationFrame(animationFrameId);
    };
  }, [interactive, isDark]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none z-0 transition-opacity duration-700"
      style={{ opacity: 0.95 }}
    />
  );
};
