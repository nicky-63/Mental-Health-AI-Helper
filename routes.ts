import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { generateChatResponse, getDailyTip } from "./services/openai";
import { insertMoodEntrySchema, messageSchema } from "@shared/schema";
import { z } from "zod";

export async function registerRoutes(app: Express): Promise<Server> {
  // Mock user ID for demo (in production, this would come from auth)
  const DEMO_USER_ID = "demo-user";

  // Chat endpoints
  app.post("/api/chat", async (req, res) => {
    try {
      const { message } = req.body;
      if (!message) {
        return res.status(400).json({ error: "Message is required" });
      }

      // Get existing chat session
      const session = await storage.getChatSession(DEMO_USER_ID);
      const previousMessages = Array.isArray(session?.messages) ? session.messages : [];

      // Generate AI response
      const aiResponse = await generateChatResponse(message, previousMessages);

      // Create new message objects
      const userMessage = {
        id: crypto.randomUUID(),
        role: "user" as const,
        content: message,
        timestamp: new Date().toISOString()
      };

      const assistantMessage = {
        id: crypto.randomUUID(),
        role: "assistant" as const,
        content: aiResponse.message,
        timestamp: new Date().toISOString(),
        isCrisisDetected: aiResponse.isCrisisDetected
      };

      // Update chat session
      const updatedMessages = [...previousMessages, userMessage, assistantMessage];
      await storage.updateChatSession(DEMO_USER_ID, updatedMessages);

      res.json({
        response: aiResponse.message,
        isCrisisDetected: aiResponse.isCrisisDetected,
        crisisResources: aiResponse.crisisResources,
        suggestedActions: aiResponse.suggestedActions
      });
    } catch (error) {
      console.error("Chat error:", error);
      res.status(500).json({ error: "Failed to generate response" });
    }
  });

  app.get("/api/chat/history", async (req, res) => {
    try {
      const session = await storage.getChatSession(DEMO_USER_ID);
      res.json({ messages: session?.messages || [] });
    } catch (error) {
      console.error("Chat history error:", error);
      res.status(500).json({ error: "Failed to get chat history" });
    }
  });

  // Mood tracking endpoints
  app.post("/api/mood", async (req, res) => {
    try {
      const moodData = insertMoodEntrySchema.parse({
        ...req.body,
        userId: DEMO_USER_ID
      });
      
      const entry = await storage.createMoodEntry(moodData);
      res.json(entry);
    } catch (error) {
      console.error("Mood entry error:", error);
      res.status(400).json({ error: "Invalid mood data" });
    }
  });

  app.get("/api/mood", async (req, res) => {
    try {
      const entries = await storage.getMoodEntries(DEMO_USER_ID);
      res.json(entries);
    } catch (error) {
      console.error("Get mood entries error:", error);
      res.status(500).json({ error: "Failed to get mood entries" });
    }
  });

  // Crisis resources endpoint (static data)
  app.get("/api/crisis-resources", async (req, res) => {
    try {
      const resources = [
        {
          title: "24/7 Crisis Support",
          description: "988 Suicide & Crisis Lifeline",
          action: "Call 988",
          href: "tel:988",
          urgent: true
        },
        {
          title: "Text Support", 
          description: "Crisis Text Line",
          action: "Text HOME to 741741",
          href: "sms:741741",
          urgent: false
        },
        {
          title: "LGBTQ+ Support",
          description: "Trevor Project",
          action: "1-866-488-7386",
          href: "tel:1-866-488-7386",
          urgent: false
        },
        {
          title: "Teen Resources",
          description: "NAMI Youth Support", 
          action: "Visit NAMI.org",
          href: "https://nami.org/Your-Journey/Kids-Teens-and-Young-Adults",
          urgent: false
        }
      ];
      res.json(resources);
    } catch (error) {
      console.error("Crisis resources error:", error);
      res.status(500).json({ error: "Failed to get crisis resources" });
    }
  });

  // Daily tip endpoint
  app.get("/api/daily-tip", async (req, res) => {
    try {
      const tip = getDailyTip();
      res.json(tip);
    } catch (error) {
      console.error("Daily tip error:", error);
      res.status(500).json({ error: "Failed to get daily tip" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
