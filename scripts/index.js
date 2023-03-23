import express from 'express';
import { exec, execFile } from 'child_process';
import dotenv from 'dotenv';
import cors from 'cors';
import fileUpload from 'express-fileupload';
import { unlinkSync } from 'fs';
import { fileURLToPath } from 'url';
import * as path from 'path';
dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(cors());
app.use(fileUpload());
app.use(express.static(path.join(__dirname, '../UploadButton')));

app.post('/demo', (req, res) => {
    const {image} = req.files;
    const standAlone = path.join(__dirname, '../src/standalone/');
    const imageDir = path.join(standAlone, image.name);
    image.mv(imageDir);
    execFile('python3', [standAlone + 'one.py', imageDir, '0'], (error, stdout, stderr) => {
        unlinkSync(imageDir);
            res.send((stdout?.toString() || '')  + '\n' +  (error?.toString() || '') /*+ '\n' + (stderr?.toString() || '')*/);
    });
});

app.use('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../UploadButton/index.html'));
});


app.listen(5000);


