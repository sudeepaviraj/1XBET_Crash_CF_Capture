const mysql = require('mysql');

const Store = (data) =>{
    const con = mysql.createConnection({
        host: process.env.DB_HOST,
        user: process.env.DB_USER,
        password: process.env.DB_PASSWORD,
        database: process.env.DB_NAME
    })
    
    con.connect((err)=>{
        if(err) throw err;
        console.log('Connected to database');
        const sql = `INSERT INTO crashes(timestamp, turn,crash_value) VALUES ([value-1],[value-2],[value-3]`
        con.query(sql, (err, result)=>{
            if(err) throw err;
            console.log('Data inserted');
        })
    })

}
Store()