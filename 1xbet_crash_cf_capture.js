const puppeteer = require("puppeteer-extra");
const launch = require("./launch");
const fs = require('fs');
const wait = (ms) => new Promise(res => setTimeout(res, ms));
const moment = require('moment')

//get WsEndpoint
async function getWsEndpoint() {
    let wsEndpoint = await launch();
    return wsEndpoint;
}

(async () => {
    const browser = await puppeteer.connect({
        browserWSEndpoint: await getWsEndpoint(),
        defaultViewport: null,
    });

    let page = await browser.newPage();
    await page.goto("https://1xbet.com/en/allgamesentrance/crash",{timeout: 0});
    await page.reload({ waitUntil: ["networkidle0", "domcontentloaded"] });


    const client = await page.target().createCDPSession()

    await client.send('Network.enable')

    client.on('Network.webSocketFrameReceived', ({ requestId, timestamp, response }) => {
        let payloadString = response.payloadData.toString('utf8');
        try {
          if (payloadString.includes('"target":"OnCrash"')) {
            payloadString = payloadString.replace(/[^\x20-\x7E]/g, '');
            const payload = JSON.parse(payloadString);
            // console.log(payload);
            const { l, f, ts } = payload.arguments[0];
            console.log(l, f, moment.unix(ts).toDate());
            const csvData = `${l},${f},${ts}\n`;
            
            fs.appendFile('data.csv', csvData, (err) => {
              if (err) throw err;
            });
          }
        } catch (error) {
          console.error('Error processing WebSocket frame:', error);
        }
      });

    while(true){
        // await page.keyboard.press("Tab");
        await wait(1000);
        // await page.keyboard.press("ArrowDown");
        await wait(1000);
    }
})();
