class Env {
    static SMTP_HOST = process.env.SMTP_HOST || '';
    static SMTP_PORT = process.env.SMTP_PORT || '';
    static SMTP_USER = process.env.SMTP_USER || '';
    static SMTP_PASSWORD = process.env.SMTP_PASSWORD || '';
    static SMTP_SECURE = process.env.SMTP_SECURE || '';
    static EMAIL_FROM = process.env.EMAIL_FROM || '';
    static SECRET_KEY = process.env.NEXTAUTH_SECRET || '';
    static APP_URL = process.env.APP_URL || '';
  }
  
  export default Env;
  