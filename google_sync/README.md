# Google Drive vault — 5 minute

Breakout Lab Python hai. Woh Google Apps Script ke andar **nahi** chal sakta.
Data **tumhare Drive** pe rahega; app Streamlit Cloud / browser pe chalegi.
Phone aur PC dono same Drive folder padhenge.

## Apps Script deploy

1. https://script.google.com → **New project**
2. `Code.gs` ki poori file paste karo (is folder se)
3. **Deploy → New deployment → Web app**
   - Execute as: **Me**
   - Who has access: **Anyone** (URL lamba hai, share mat karna)
4. **Deploy** → URL copy (`https://script.google.com/macros/s/…/exec`)
5. Breakout Lab sidebar → **Data folder** → Google vault URL → paste → **Save & pull from Drive**
6. Drive me folder **Breakout Lab** banega — har book ek JSON

Pehli baar Google **permission** maangega (Drive). Allow.

## Phir

- Har buy/sell Drive pe save
- Naya phone: same Streamlit link + same Google account nahi chahiye app ko — vault URL se Drive padhega
- Sheet/Drive khol ke JSON backup bhi dekh sakte ho
