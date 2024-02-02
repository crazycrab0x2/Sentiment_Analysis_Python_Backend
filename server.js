const express = require('express');
const app = express();
const formidable = require('formidable');
const path = require('node:path');
const fs = require('node:fs');
const throttle = require('express-throttle-bandwidth');
const sqlite3 = require('sqlite3').verbose();
const Set = require('set');
// const db = new sqlite3.Database('mydb.sqlite');

const
	port = process.env.PORT || 4000,
	folder = path.join(__dirname, 'uploads')

if (!fs.existsSync(folder)) {
	fs.mkdirSync(folder)
}

app.set('port', port)
app.use(throttle(1024 * 128)) // throttling bandwidth

app.use((req, res, next) => {
	res.header('Access-Control-Allow-Origin', '*')
	res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept')
	next()
})

app.post('/upload', (req, res) => {
	const form = new formidable.IncomingForm()

	form.uploadDir = folder
	form.parse(req, (_, fields, files) => {
	  console.log('\n-----------')
	  console.log('Fields', fields)
	  console.log('Received:', Object.keys(files))
	  console.log()
	  res.send('file uploaded')
	})
	// const oldFiles = fs.readdirSync(folder);
	// const oldSet = new Set(oldFiles);
	// let newFileName;
	// const form = new formidable.IncomingForm()
	// form.uploadDir = folder
	// form.parse(req, (_, fields, files) => {
	// 	newFileName = Object.keys(files)[0]
	// 	newFiles = fs.readdirSync(folder);
	// 	const newSet = new Set(newFiles)
	// 	const newfile = oldSet.difference(newSet)
	// 	console.log(newfile.get()[0])
	// 	fs.rename(`${folder}/${newfile.get()[0]}`, `${folder}/${newFileName}`, ()=>{})
	// 	res.send('file saved!')
	// })
})

app.listen(port, () => {
	console.log('\nUpload server running on http://localhost:' + port)
})