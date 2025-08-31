import { type ChatSession, type InsertChatSession, type MoodEntry, type InsertMoodEntry } from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  // Chat Sessions
  getChatSession(userId: string): Promise<ChatSession | undefined>;
  createChatSession(session: InsertChatSession): Promise<ChatSession>;
  updateChatSession(userId: string, messages: any[]): Promise<ChatSession>;

  // Mood Entries
  getMoodEntries(userId: string): Promise<MoodEntry[]>;
  createMoodEntry(entry: InsertMoodEntry): Promise<MoodEntry>;
  getRecentMoodEntries(days: number): Promise<MoodEntry[]>;
}

export class MemStorage implements IStorage {
  private chatSessions: Map<string, ChatSession>;
  private moodEntries: Map<string, MoodEntry>;

  constructor() {
    this.chatSessions = new Map();
    this.moodEntries = new Map();
  }

  // Chat Sessions
  async getChatSession(userId: string): Promise<ChatSession | undefined> {
    return Array.from(this.chatSessions.values()).find(
      (session) => session.userId === userId
    );
  }

  async createChatSession(insertSession: InsertChatSession): Promise<ChatSession> {
    const id = randomUUID();
    const session: ChatSession = { 
      id, 
      userId: insertSession.userId || "demo-user",
      messages: insertSession.messages || [],
      createdAt: new Date() 
    };
    this.chatSessions.set(id, session);
    return session;
  }

  async updateChatSession(userId: string, messages: any[]): Promise<ChatSession> {
    const session = await this.getChatSession(userId);
    if (session) {
      session.messages = messages;
      this.chatSessions.set(session.id, session);
      return session;
    }
    // Create new session if none exists
    return this.createChatSession({ userId: userId || "demo-user", messages });
  }

  // Mood Entries
  async getMoodEntries(userId: string): Promise<MoodEntry[]> {
    return Array.from(this.moodEntries.values())
      .filter((entry) => entry.userId === userId)
      .sort((a, b) => (b.createdAt?.getTime() || 0) - (a.createdAt?.getTime() || 0));
  }

  async createMoodEntry(insertEntry: InsertMoodEntry): Promise<MoodEntry> {
    const id = randomUUID();
    const entry: MoodEntry = { 
      id, 
      userId: insertEntry.userId || "demo-user",
      mood: insertEntry.mood,
      notes: insertEntry.notes || null,
      createdAt: new Date() 
    };
    this.moodEntries.set(id, entry);
    return entry;
  }

  async getRecentMoodEntries(days: number): Promise<MoodEntry[]> {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - days);
    
    return Array.from(this.moodEntries.values())
      .filter((entry) => (entry.createdAt?.getTime() || 0) >= cutoffDate.getTime());
  }

}

export const storage = new MemStorage();
