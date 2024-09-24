const cheerio = require('cheerio');
const axios = require('axios');
const fs = require('fs');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;

// Path to CSV file
const csvFilePath = 'news_data1.csv';

// Create or append to the CSV file depending on whether the file exists
const csvWriter = createCsvWriter({
    path: csvFilePath,
    header: [
        { id: 'title', title: 'Title' },
        { id: 'href', title: 'Links' }
    ],
    append: fs.existsSync(csvFilePath) // Append if the file exists, else write headers
});

// Function to extract and save data to CSV
async function scrapeAndSave(url, siteName) {
    try {
        let newsData = [];
        const response = await axios.get(url);
        const $ = cheerio.load(response.data);
        $('a').each(function (index, element) {
            const title = $(this).attr('title') || $(this).text(); // Fallback to text if 'title' attribute is missing
            const href = $(this).attr('href');
            if (title && href) {
                newsData.push({ title, href });
            }
        });

        if (newsData.length > 0) {
            // Append data to CSV
            await csvWriter.writeRecords(newsData);
            console.log(`${siteName} data saved to ${csvFilePath}`);
        }
    } catch (error) {
        console.log(`Error scraping ${siteName}:`, error);
    }
}

// URL of the websites to scrape
//const indiaTodayUrl = 'https://www.indiatoday.in/india';
//const ndtvUrl = 'https://www.ndtv.com/india#pfrom=home-ndtv_mainnavigation';
//const republicUrl = 'https://www.republicworld.com/india';
const timesNowUrl = 'https://www.timesnownews.com/india';
//const news18IndiaUrl = 'https://www.news18.com/india';
//const zeeNewsUrl = 'https://zeenews.india.com/latest-news';

// Scrape and save data
// scrapeAndSave(indiaTodayUrl, 'IndiaToday');
// scrapeAndSave(ndtvUrl, 'NDTV');
// scrapeAndSave(republicUrl, 'RepublicTv');
scrapeAndSave(timesNowUrl, 'Times Now');
// scrapeAndSave(news18IndiaUrl, 'News18India');
// scrapeAndSave(zeeNewsUrl, 'Zee News');
