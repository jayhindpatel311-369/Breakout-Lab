/**
 * Breakout Lab — Google Drive vault
 *
 * Deploy: Deploy → New deployment → Web app
 *   Execute as: Me
 *   Who has access: Anyone  (URL secret hai; Streamlit Cloud isse call karega)
 * Copy the URL into Breakout Lab → Data folder → Google vault URL
 *
 * Creates a Drive folder "Breakout Lab" and stores one JSON file per book.
 */

function getFolder_() {
  var props = PropertiesService.getScriptProperties();
  var id = props.getProperty("FOLDER_ID");
  if (id) {
    try {
      return DriveApp.getFolderById(id);
    } catch (e) {}
  }
  var f = DriveApp.createFolder("Breakout Lab");
  props.setProperty("FOLDER_ID", f.getId());
  return f;
}

function json_(obj) {
  var text = typeof obj === "string" ? obj : JSON.stringify(obj);
  return ContentService.createTextOutput(text).setMimeType(ContentService.MimeType.JSON);
}

function safeName_(name) {
  return String(name || "book").replace(/[^\w \-]+/g, "").replace(/\s+/g, "_").slice(0, 80) || "book";
}

function doGet(e) {
  var folder = getFolder_();
  var params = (e && e.parameter) || {};
  var want = String(params.name || "").trim();
  if (want) {
    var files = folder.getFilesByName(safeName_(want) + ".json");
    if (!files.hasNext()) {
      return json_({ ok: false, error: "not found" });
    }
    return json_(files.next().getBlob().getDataAsString());
  }
  var books = {};
  var it = folder.getFiles();
  while (it.hasNext()) {
    var f = it.next();
    if (/\.json$/i.test(f.getName())) {
      books[f.getName().replace(/\.json$/i, "")] = f.getBlob().getDataAsString();
    }
  }
  return json_({ ok: true, books: books, folder: folder.getUrl() });
}

function doPost(e) {
  if (!e || !e.postData || !e.postData.contents) {
    return json_({ ok: false, error: "empty body" });
  }
  var body = JSON.parse(e.postData.contents);
  var name = safeName_(body.name);
  var data = typeof body.book === "string" ? body.book : JSON.stringify(body.book);
  var folder = getFolder_();
  var files = folder.getFilesByName(name + ".json");
  if (files.hasNext()) {
    files.next().setContent(data);
  } else {
    folder.createFile(name + ".json", data, MimeType.PLAIN_TEXT);
  }
  return json_({ ok: true, name: name, folder: folder.getUrl() });
}
