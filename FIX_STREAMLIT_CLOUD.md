# Fix: supabase ModuleNotFoundError on Streamlit Cloud

**Problem**:
```
ModuleNotFoundError: This app has encountered an error...
from config import init_supabase, supabase_config
```

**Cause**: The Supabase package is not installed in your Streamlit Cloud environment even though `requirements.txt` lists it.

---

## Solution: Reboot Your Streamlit Cloud App

Streamlit Cloud sometimes doesn't rebuild the environment automatically when you update `requirements.txt`. Follow these steps:

### Step 1: Go to Streamlit Cloud
1. Visit https://share.streamlit.io
2. Find your app in the list
3. Click on the three dots **(...) menu** in the top right

### Step 2: Manage App
1. Click **"Manage app"** from the dropdown menu

### Step 3: Reboot the App
1. Scroll down to **"Reboot app"** section
2. Click the **"Reboot"** button
3. **Wait 2-5 minutes** for the app to rebuild and restart
   - This is important - it needs time to reinstall all dependencies including `supabase`

### Step 4: Verify
1. The app should reload automatically when ready
2. You should see **"Your app is running"** message (green indicator)
3. If it still fails, proceed to the troubleshooting section below

**This should fix the error in most cases.** The reboot forces Streamlit Cloud to rebuild the Python environment and install all dependencies from `requirements.txt`.

---

## Verification: Check Your requirements.txt

Make sure your `requirements.txt` file has these lines (in this exact order):

```
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.15.0
numpy>=1.24.0
openpyxl>=3.1.0
matplotlib>=3.7.0
supabase>=2.23.0
python-dotenv>=1.0.0
```

The file should have **8 lines total** with the last two being:
- Line 7: `supabase>=2.23.0`
- Line 8: `python-dotenv>=1.0.0`

If the file is missing these lines, update it and push to GitHub again.

---

## Check: Verify Secrets in Streamlit Cloud

Your app needs Supabase credentials. In Streamlit Cloud:

1. Go to **"Manage app"** → **"Settings"**
2. Scroll to **"Secrets"** section
3. Add these two secrets:
   ```
   SUPABASE_URL = "https://your-project-id.supabase.co"
   SUPABASE_KEY = "your-anon-key"
   ```
4. **Save** the changes

---

## Step-by-Step Troubleshooting

### If Reboot Alone Doesn't Work:

**Option 1: Force a Clean Rebuild**
1. Go to Streamlit Cloud app settings
2. Click **"Reboot"** button
3. Wait for complete rebuild (2-5 minutes)

**Option 2: Clear Cache and Redeploy**
1. Go to app settings
2. Click **"Advanced settings"**
3. Look for **"Clear cache"** or **"Reset app"**
4. Confirm and wait for rebuild

**Option 3: Re-push to GitHub**
```bash
# Make a small change to trigger rebuild
echo "# Updated" >> streamlit_app.py
git add streamlit_app.py
git commit -m "Trigger Streamlit rebuild"
git push origin main
```
Streamlit Cloud should automatically redeploy.

---

## Expected Error Messages (and what they mean)

### Error: `ModuleNotFoundError: No module named 'supabase'`
**Solution**: Reboot the app (Step 3 above)

### Error: `ModuleNotFoundError: No module named 'dotenv'`
**Solution**: Reboot the app or update requirements.txt with `python-dotenv>=1.0.0`

### Error: `Supabase가 설정되지 않았습니다` (Supabase not configured)
**Solution**: Add secrets in Streamlit Cloud settings (Step 3 above)

### Error: Module imports but app won't load
**Solution**: Check that secrets are correctly set in Streamlit Cloud

---

## Verification Checklist

After rebooting, verify:

- [ ] `requirements.txt` has `supabase>=2.23.0` on line 7
- [ ] `requirements.txt` has `python-dotenv>=1.0.0` on line 8
- [ ] Streamlit Cloud secrets include `SUPABASE_URL`
- [ ] Streamlit Cloud secrets include `SUPABASE_KEY`
- [ ] App has been rebooted in the last 2 minutes
- [ ] App shows "Your app is running" (green indicator)
- [ ] No error messages in the main panel

---

## Quick Reference: All Required Secrets

For **Streamlit Cloud**, add these secrets:

```
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "eyJhbGciOi..." (your anon key)
```

For **Local Development**, create `.env`:
```
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-key
```

---

## Still Not Working?

If the app still shows errors after rebooting:

1. **Check logs**: In Streamlit Cloud, click "Manage app" → "Logs" to see error details
2. **Verify requirements.txt**: Make sure it has exactly 8 lines with supabase on line 7
3. **Verify secrets**: Go to Settings → Secrets and confirm both `SUPABASE_URL` and `SUPABASE_KEY` are set
4. **Try hard refresh**: In browser, press Ctrl+Shift+R (or Cmd+Shift+R on Mac) to clear cache
5. **Wait longer**: Sometimes it takes 3-5 minutes to fully deploy

---

## Contact Support

If you still have issues after trying all steps above:

1. Check Streamlit's documentation: https://docs.streamlit.io/deploy/tutorials/deploy-a-python-app
2. Review [GITHUB_SETUP_GUIDE.md](./GITHUB_SETUP_GUIDE.md) for detailed deployment instructions
3. Check Streamlit Cloud logs for specific error messages

---

**Summary**:
- **Most likely fix**: Click "Reboot" in Streamlit Cloud app settings
- **Time to fix**: 2-5 minutes
- **Should work after**: Reboot completes and app reloads
